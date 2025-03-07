import json
import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import re
from json_repair import repair_json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入原生 API 客户端
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai

# 模型价格配置（每1000个token的价格，单位为美元）
MODEL_PRICE_CONFIG = {
    # OpenAI 模型
    "gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.0006
    },
    "gpt-3.5-turbo": {
        "input": 0.0000005,
        "output": 0.0000015
    },
    "gpt-4": {
        "input": 0.00003,
        "output": 0.00006
    },
    
    # Anthropic 模型
    "claude-3-5-sonnet-20241022": {
        "input": 0.003,
        "output": 0.015
    },
    "claude-3-sonnet-20240229": {
        "input": 0.003,
        "output": 0.015
    },
    "claude-3-opus-20240229": {
        "input": 0.015,
        "output": 0.075
    },
    
    # Google 模型
    "gemini-2.0-flash": {
        "input": 0.0001,
        "output": 0.0004
    },
    "gemini-2.0-flash-lite": {
        "input": 0.000075,
        "output": 0.0003
    },
    "gemini-pro": {
        "input": 0.000125,
        "output": 0.000375
    }
}

# 定义模型提供商枚举
class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"

# LLM 响应类
class LLMResponse:
    def __init__(
        self,
        content: Dict[str, Any],
        model_name: str,
        usage: Dict[str, int],
        raw_response: Any = None,
        timestamp: Optional[datetime] = None,
        processing_time: float = 0.0,
        total_tokens: Optional[int] = None,
        model_config: Optional[dict] = None,
        cost: float = 0.0,
        response_valid: bool = True  # 添加此参数
    ):
        self.content = content
        self.model_name = model_name
        self.usage = usage
        self.raw_response = raw_response
        self.timestamp = timestamp or datetime.utcnow()
        self.processing_time = processing_time
        self.total_tokens = total_tokens
        self.model_config = model_config
        self.cost = cost
        self.response_valid = response_valid  # 存储响应是否有效

# Schema适配器类
class SchemaAdapter:
    """处理不同LLM提供商的schema适配"""
    
    @staticmethod
    def adapt_schema(schema: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """返回适合特定提供商API的schema对象"""
        if provider == "openai":
            # OpenAI 使用标准 JSON schema
            return schema
        elif provider == "anthropic":
            # Anthropic 通过提示传递 schema，这里返回空字典
            return {}
        elif provider == "google":
            # Google 没有专门的 schema 参数
            return {}
        else:
            return schema
    
    @staticmethod
    def format_prompt(prompt: str, schema: Dict[str, Any], provider: str) -> str:
        """根据提供商将schema集成到提示中"""
        if provider == "anthropic":
            # Anthropic 使用 response_schema
            schema_str = json.dumps(schema, ensure_ascii=False)
            return f"{prompt}\n\nresponse_schema {schema_str}"
        elif provider == "google":
            # Google 需要在提示中描述结构
            schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
            return f"{prompt}\n\n请按照以下JSON结构返回结果:\n{schema_str}"
        else:
            # OpenAI 不需要在提示中包含schema
            return prompt

# 计算API调用成本的函数
def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    计算API调用成本
    
    Args:
        model_name: 模型名称
        input_tokens: 输入token数量
        output_tokens: 输出token数量
        
    Returns:
        成本（美元）
    """
    if model_name not in MODEL_PRICE_CONFIG:
        # 如果找不到精确匹配，尝试查找前缀匹配
        for key in MODEL_PRICE_CONFIG:
            if model_name.startswith(key):
                model_name = key
                break
        else:
            # 如果仍然找不到匹配，返回0
            return 0.0
    
    # 计算成本
    price_config = MODEL_PRICE_CONFIG[model_name]
    input_cost = (input_tokens / 1000) * price_config["input"]
    output_cost = (output_tokens / 1000) * price_config["output"]
    
    return input_cost + output_cost

# 基础 LLM 抽象类
class BaseLLM(ABC):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_response: Optional[LLMResponse] = None
        self.provider = None  # 子类应设置此属性
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """异步生成回复"""
        pass
    
    def _format_prompt_with_schema(self, prompt: str, schema: Dict[str, Any]) -> str:
        """根据提供商格式化包含schema的提示"""
        return SchemaAdapter.format_prompt(prompt, schema, self.provider)
    
    def _adapt_schema_for_api(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """根据提供商调整schema用于API调用"""
        return SchemaAdapter.adapt_schema(schema, self.provider)

# OpenAI LLM 实现
class OpenAILLM(BaseLLM):
    async def generate(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        try:
            start_time = time.perf_counter()
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # 使用 SchemaAdapter 格式化提示
            formatted_prompt = SchemaAdapter.format_prompt(prompt, schema, "openai")
            
            # 添加 "json" 关键字到提示中如果使用 response_format
            if schema:
                formatted_prompt = f"{formatted_prompt}\n请以JSON格式回复，确保包含所有必需字段: {', '.join(schema.get('required', []))}。"
                
            messages.append({"role": "user", "content": formatted_prompt})
            
            # 构建请求参数
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                **kwargs
            }
            
            # 只有当 max_tokens 不为 None 时才添加到参数中
            if self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens
            
            # 添加 response_format 参数，使用 json_object 而不是 json
            if schema:
                request_params["response_format"] = {"type": "json_object"}
            
            # 发送请求
            completion = await client.chat.completions.create(**request_params)
            
            # 解析响应
            raw_content = completion.choices[0].message.content
            response_valid = True  # 默认假设响应有效
            
            try:
                # 尝试解析 JSON
                output_content = json.loads(raw_content)
                
                # 验证响应是否包含所有必需字段
                if schema and "required" in schema:
                    required_fields = schema["required"]
                    missing_fields = [field for field in required_fields if field not in output_content]
                    response_valid = len(missing_fields) == 0
                    
                    # 如果缺少字段，添加错误信息
                    if not response_valid:
                        output_content["_missing_fields"] = missing_fields
                
            except json.JSONDecodeError:
                # 如果解析失败，尝试修复 JSON
                try:
                    fixed_json = repair_json(raw_content)
                    output_content = json.loads(fixed_json)
                    
                    # 验证修复后的JSON
                    if schema and "required" in schema:
                        required_fields = schema["required"]
                        missing_fields = [field for field in required_fields if field not in output_content]
                        response_valid = len(missing_fields) == 0
                        
                        # 如果缺少字段，添加错误信息
                        if not response_valid:
                            output_content["_missing_fields"] = missing_fields
                    
                except:
                    # 如果仍然失败，返回原始内容
                    output_content = {"raw_response": raw_content}
                    response_valid = False
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # 确保正确获取 token 使用情况
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
            
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            response = LLMResponse(
                content=output_content,
                model_name=self.model_name,
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": completion.usage.total_tokens
                },
                raw_response=completion,
                timestamp=datetime.utcnow(),
                processing_time=processing_time,
                total_tokens=completion.usage.total_tokens,
                model_config={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "provider": "openai"
                },
                cost=cost,
                response_valid=response_valid
            )
            
            self.last_response = response
            return response
            
        except Exception as e:
            raise Exception(f"OpenAI API 调用失败: {str(e)}")


# Anthropic LLM 实现
class AnthropicLLM(BaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.provider = "anthropic"
    
    async def generate(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        try:
            start_time = time.perf_counter()
            
            # 将 schema 集成到提示中
            formatted_prompt = self._format_prompt_with_schema(prompt, schema)
            
            messages = [{"role": "user", "content": formatted_prompt}]
            
            # 添加系统提示
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
                
            # 增加其他参数
            for key, value in kwargs.items():
                request_params[key] = value
                
            # 只有当 max_tokens 不为 None 时才添加到参数中
            if self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens
            
            response = await self.client.messages.create(**request_params)
            
            raw_content = response.content[0].text
            try:
                # 尝试直接解析
                output_content = json.loads(raw_content)
            except json.JSONDecodeError:
                # 尝试查找JSON块
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', raw_content, re.DOTALL)
                if json_match:
                    try:
                        output_content = json.loads(json_match.group(1))
                    except:
                        # 尝试修复JSON
                        output_content = json.loads(repair_json(json_match.group(1)))
                else:
                    # 尝试修复 JSON
                    output_content = json.loads(repair_json(raw_content))
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # 计算成本
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            llm_response = LLMResponse(
                content=output_content,
                model_name=self.model_name,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                raw_response=response,
                timestamp=datetime.utcnow(),
                processing_time=processing_time,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                model_config={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "provider": "anthropic"
                },
                cost=cost
            )
            
            self.last_response = llm_response
            return llm_response
            
        except Exception as e:
            raise Exception(f"Anthropic API 调用失败: {str(e)}")

# Google LLM 实现
class GoogleLLM(BaseLLM):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ):
        super().__init__(model_name, api_key, temperature, max_tokens)
        self.provider = "google"
    
    async def generate(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        try:
            start_time = time.perf_counter()
            
            # 初始化 client 时提供 API key
            client = genai.Client(api_key=self.api_key)
            
            # 使用 SchemaAdapter 格式化提示
            formatted_prompt = SchemaAdapter.format_prompt(prompt, schema, "google")
            
            # 构建完整提示
            full_prompt = formatted_prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{formatted_prompt}"
            
            # 简化调用，只传递必要参数
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            
            # 获取响应文本
            raw_content = response.text
            response_valid = True
            
            try:
                # 尝试解析 JSON
                output_content = json.loads(raw_content)
                
                # 验证响应是否包含所有必需字段
                if schema and "required" in schema:
                    required_fields = schema["required"]
                    missing_fields = [field for field in required_fields if field not in output_content]
                    response_valid = len(missing_fields) == 0
                    
                    # 如果缺少字段，添加错误信息
                    if not response_valid:
                        output_content["_missing_fields"] = missing_fields
            except json.JSONDecodeError:
                # 尝试修复 JSON
                try:
                    fixed_json = repair_json(raw_content)
                    output_content = json.loads(fixed_json)
                    response_valid = True
                except:
                    # 如果仍然失败，返回原始内容
                    output_content = {"raw_response": raw_content}
                    response_valid = False
            
            # 计算处理时间（毫秒）
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # 估算token数量
            input_tokens = len(full_prompt.split()) // 3 * 4
            output_str = json.dumps(output_content, ensure_ascii=False)
            output_tokens = len(output_str.split()) // 3 * 4
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
            # 计算成本
            cost = calculate_cost(self.model_name, input_tokens, output_tokens)
            
            # 创建响应对象
            response_obj = LLMResponse(
                content=output_content,
                model_name=self.model_name,
                usage=usage,
                raw_response=response,
                timestamp=datetime.utcnow(),
                processing_time=processing_time,
                total_tokens=usage.get("total_tokens", 0),
                model_config={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "provider": "google"
                },
                cost=cost,
                response_valid=response_valid
            )
            
            self.last_response = response_obj
            return response_obj
            
        except Exception as e:
            raise Exception(f"Google API 调用失败: {str(e)}")

# LLM 工厂类
class LLMFactory:
    @staticmethod
    def create_llm(
        provider: str,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> BaseLLM:
        """
        创建 LLM 实例
        
        Args:
            provider: 提供商名称 ("openai", "anthropic", "google")
            model_name: 模型名称，如果为 None 则使用默认模型
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            BaseLLM 实例
        """
        # 获取环境变量中的API密钥
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        # 尝试多个可能的Google API密钥环境变量名
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        provider_map = {
            "openai": {
                "class": OpenAILLM,
                "default_model": "gpt-4o-mini",
                "api_key": openai_api_key
            },
            "anthropic": {
                "class": AnthropicLLM,
                "default_model": "claude-3-5-sonnet-20241022",
                "api_key": anthropic_api_key
            },
            "google": {
                "class": GoogleLLM,
                "default_model": "gemini-2.0-flash",
                "api_key": google_api_key
            }
        }
        
        if provider not in provider_map:
            raise ValueError(f"不支持的提供商: {provider}")
        
        provider_info = provider_map[provider]
        
        # 检查API密钥是否存在
        if not provider_info["api_key"]:
            raise ValueError(f"缺少 {provider} 的API密钥，请在.env文件中设置")
        
        return provider_info["class"](
            model_name=model_name or provider_info["default_model"],
            api_key=provider_info["api_key"],
            temperature=temperature,
            max_tokens=max_tokens
        )