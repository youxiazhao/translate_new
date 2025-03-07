import json
import os
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from enum import Enum
from datetime import datetime
import langcodes  # 导入langcodes库
import pandas as pd
import logging
import sys
# 尝试导入language_data包，用于获取语言名称
try:
    import language_data
    has_language_data = True
except ImportError:
    has_language_data = False
    print("警告: language_data包未安装。将无法获取完整的语言名称信息。可使用 'pip install language_data' 安装。")

# 加载环境变量
load_dotenv()

# 导入 LLM 工厂和响应类
from llm_client import LLMFactory, LLMResponse

# 导入提示模板
from prompts import (
    TRANSLATION_PROMPT_TEMPLATE,
    EVALUATION_PROMPT_TEMPLATE,
    RETRY_TRANSLATION_PROMPT_TEMPLATE,
    get_schema_for_prompt_type
)

# 配置日志
def setup_logging(level=logging.DEBUG):
    """设置日志配置"""
    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 创建文件处理器
    file_handler = logging.FileHandler('translation_pipeline.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除现有处理器
    logger.handlers = []
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 禁用第三方库的详细日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logger

# 初始化日志
logger = setup_logging()

# 增加语言代码辅助函数
def normalize_language_code(language_code_or_name: str) -> str:
    """
    标准化语言代码或将语言名称转换为标准语言代码
    
    Args:
        language_code_or_name: 语言代码或名称
        
    Returns:
        标准化的语言代码
    """
    try:
        # 尝试作为语言代码处理
        return langcodes.standardize_tag(language_code_or_name)
    except (ValueError, TypeError) as e:
        try:
            # 尝试作为语言名称处理
            return str(langcodes.find(language_code_or_name))
        except (LookupError, ValueError, TypeError) as e:
            # 无法识别，返回原始输入
            print(f"警告: 无法识别语言代码或名称 '{language_code_or_name}'，使用默认值 'und'")
            return "und"  # 未定义语言代码

def get_language_display_name(language_code: str, display_language: str = "en") -> str:
    """
    获取语言的显示名称
    
    Args:
        language_code: 语言代码
        display_language: 显示名称使用的语言（默认英语）
        
    Returns:
        语言的显示名称
    """
    if not has_language_data:
        # 如果没有language_data，返回原始代码
        return language_code
    
    try:
        language = langcodes.Language.get(language_code)
        return language.display_name(display_language)
    except:
        return language_code

def get_language_autonym(language_code: str) -> str:
    """
    获取语言的自名（语言在该语言中的称呼）
    
    Args:
        language_code: 语言代码
        
    Returns:
        语言的自名
    """
    if not has_language_data:
        # 如果没有language_data，返回原始代码
        return language_code
    
    try:
        language = langcodes.Language.get(language_code)
        return language.autonym()
    except:
        return language_code

def is_valid_language_code(language_code: str) -> bool:
    """
    检查语言代码是否有效
    
    Args:
        language_code: 语言代码
        
    Returns:
        语言代码是否有效
    """
    return langcodes.tag_is_valid(language_code)

# 运行详情模型
class ModelRunDetails(BaseModel):
    model_name: str = Field(description="使用的模型名称")
    provider: str = Field(description="提供商名称")
    temperature: float = Field(description="使用的温度参数")
    prompt: str = Field(description="发送给模型的提示")
    system_prompt: Optional[str] = Field(description="系统提示（如果使用）")
    input_tokens: int = Field(description="输入token数量")
    output_tokens: int = Field(description="输出token数量")
    total_tokens: int = Field(description="总token数量")
    cost: float = Field(description="API调用成本（美元）")
    processing_time_ms: float = Field(description="处理时间（毫秒）")
    timestamp: datetime = Field(description="完成时间")
    response: Dict[str, Any] = Field(description="模型响应内容")

# 评估指标模型
class EvaluationScores(BaseModel):
    language_accuracy: float = Field(description="语言准确性评分 (0-10)")
    semantic_accuracy: float = Field(description="语义准确性评分 (0-10)")
    fluency: float = Field(description="流畅性评分 (0-10)")
    style: float = Field(description="风格评分 (0-10)")
    terminology: float = Field(description="术语准确性评分 (0-10)")

# 评估结果模型
class EvaluationResult(BaseModel):
    scores: EvaluationScores = Field(description="各维度评分")
    total_score: float = Field(description="总评分 (0-10)")
    feedback: str = Field(description="详细评估反馈")
    run_details: Optional[ModelRunDetails] = Field(None, description="评估模型运行详情")

# 翻译尝试模型 - 新增语言代码字段
class TranslationAttempt(BaseModel):
    attempt: int = Field(description="尝试次数")
    translation: str = Field(description="翻译结果")
    source_language: str = Field(description="源语言")
    source_language_code: str = Field(description="标准化的源语言代码")
    source_language_name: str = Field(description="源语言名称")
    explanation: Optional[str] = Field(None, description="翻译解释")
    evaluation: Optional[EvaluationResult] = Field(None, description="评估结果")
    run_details: ModelRunDetails = Field(description="模型运行详情")

# 翻译结果模型 - 新增语言代码字段
class TranslationResult(BaseModel):
    original_text: str = Field(description="原始文本")
    target_language: str = Field(description="目标语言")
    target_language_code: str = Field(description="标准化的目标语言代码")
    target_language_name: str = Field(description="目标语言名称（英文）")
    target_language_autonym: str = Field(description="目标语言名称（自名）")
    final_translation: str = Field(description="最终翻译结果")
    source_language: str = Field(description="源语言")
    source_language_code: str = Field(description="标准化的源语言代码")
    source_language_name: str = Field(description="源语言名称（英文）")
    source_language_autonym: str = Field(description="源语言名称（自名）")
    attempts: int = Field(description="尝试次数")
    attempts_details: List[TranslationAttempt] = Field(description="每次尝试的详情")
    total_tokens: int = Field(description="总token数量")
    total_cost: float = Field(description="总成本（美元）")
    total_time_ms: float = Field(description="总处理时间（毫秒）")
    final_score: float = Field(description="最终评分 (0-10)")

# 创建一个具有适当属性的对象，用于格式化字符串
class FeedbackObject:
    def __init__(self, total_score, scores, feedback):
        self.total_score = total_score
        self.scores = scores
        self.feedback = feedback

# 异步翻译 Pipeline 类
class AsyncTranslationPipeline:
    def __init__(
        self,
        translator_provider: str = "openai",
        translator_model: Optional[str] = None,
        evaluator_provider: str = "anthropic",
        evaluator_model: Optional[str] = None,
        retry_translator_provider: str = "google",
        retry_translator_model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        max_attempts: int = 3,
        min_acceptable_score: float = 8.0,
        system_prompt: Optional[str] = None,
        display_language: str = "en"  # 用于显示语言名称的语言
    ):
        """
        初始化异步翻译 Pipeline
        
        Args:
            translator_provider: 初始翻译使用的提供商
            translator_model: 初始翻译使用的模型
            evaluator_provider: 评估使用的提供商
            evaluator_model: 评估使用的模型
            retry_translator_provider: 重试翻译使用的提供商
            retry_translator_model: 重试翻译使用的模型
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            max_attempts: 最大尝试次数
            min_acceptable_score: 最低可接受评分
            system_prompt: 系统提示（如果使用）
            display_language: 用于显示语言名称的语言代码
        """
        self.translator_provider = translator_provider
        self.translator_model = translator_model
        self.evaluator_provider = evaluator_provider
        self.evaluator_model = evaluator_model
        self.retry_translator_provider = retry_translator_provider
        self.retry_translator_model = retry_translator_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.min_acceptable_score = min_acceptable_score
        self.system_prompt = system_prompt
        self.display_language = normalize_language_code(display_language)
        
        # 创建 LLM 实例
        self.translator = LLMFactory.create_llm(
            provider=translator_provider,
            model_name=translator_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.evaluator = LLMFactory.create_llm(
            provider=evaluator_provider,
            model_name=evaluator_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.retry_translator = LLMFactory.create_llm(
            provider=retry_translator_provider,
            model_name=retry_translator_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def translate(
        self,
        text: str,
        target_language: str
    ) -> TranslationResult:
        """
        翻译文本并评估质量，如果需要则重试
        
        Args:
            text: 要翻译的文本
            target_language: 目标语言（代码或名称）
            
        Returns:
            TranslationResult: 翻译结果
        """
        start_time = time.perf_counter()
        
        # 标准化目标语言
        target_language_code = normalize_language_code(target_language)
        target_language_name = get_language_display_name(target_language_code, self.display_language)
        target_language_autonym = get_language_autonym(target_language_code)
        
        # 使用语言名称而不是代码来给LLM提供更好的上下文
        target_language_for_prompt = target_language_name if has_language_data else target_language
        
        # 初始化结果
        result = TranslationResult(
            original_text=text,
            target_language=target_language,
            target_language_code=target_language_code,
            target_language_name=target_language_name,
            target_language_autonym=target_language_autonym,
            final_translation="",
            source_language="",
            source_language_code="",
            source_language_name="",
            source_language_autonym="",
            attempts=0,
            attempts_details=[],
            total_tokens=0,
            total_cost=0.0,
            total_time_ms=0.0,
            final_score=0.0
        )
        
        # 获取翻译 schema
        translation_schema = get_schema_for_prompt_type("translation")
        
        # 在初始翻译部分
        logger.info(f"开始翻译文本: '{text[:50]}...' 到 {target_language}")
        logger.debug(f"目标语言标准化: {target_language} -> {target_language_code} ({target_language_name})")
        
        # 初始翻译
        print(f"开始翻译文本到 {target_language_name} ({target_language_code})...")
        translation_response = await self.translator.generate(
            prompt=TRANSLATION_PROMPT_TEMPLATE.format(
                text=text,
                target_language=target_language_for_prompt
            ),
            schema=translation_schema,
            system_prompt=self.system_prompt
        )
        
        # 解析翻译结果
        translation_content = translation_response.content
        translation = translation_content.get("translated_text", "")
        source_language = translation_content.get("source_language", "未知")
        explanation = translation_content.get("explanation", None)
        
        # 标准化源语言
        source_language_code = normalize_language_code(source_language)
        source_language_name = get_language_display_name(source_language_code, self.display_language)
        source_language_autonym = get_language_autonym(source_language_code)
        
        # 创建翻译尝试记录
        first_attempt = TranslationAttempt(
            attempt=1,
            translation=translation,
            source_language=source_language,
            source_language_code=source_language_code,
            source_language_name=source_language_name,
            explanation=explanation,
            run_details=ModelRunDetails(
                model_name=translation_response.model_name,
                provider=self.translator_provider,
                temperature=self.temperature,
                prompt=TRANSLATION_PROMPT_TEMPLATE.format(
                    text=text,
                    target_language=target_language_for_prompt
                ),
                system_prompt=self.system_prompt,
                input_tokens=translation_response.usage.get("input_tokens", 0),
                output_tokens=translation_response.usage.get("output_tokens", 0),
                total_tokens=translation_response.total_tokens,
                cost=translation_response.cost,
                processing_time_ms=translation_response.processing_time,
                timestamp=translation_response.timestamp,
                response=translation_content
            )
        )
        
        # 更新结果
        result.attempts += 1
        result.attempts_details.append(first_attempt)
        result.total_tokens += translation_response.total_tokens
        result.total_cost += translation_response.cost
        result.source_language = source_language
        result.source_language_code = source_language_code
        result.source_language_name = source_language_name
        result.source_language_autonym = source_language_autonym
        
        # 检查源语言和目标语言是否相同
        if langcodes.tag_distance(source_language_code, target_language_code) == 0:
            print(f"源语言与目标语言相同 ({source_language_name})，返回原文。")
            result.final_translation = text
            result.final_score = 10.0
            result.total_time_ms = (time.perf_counter() - start_time) * 1000
            return result
        
        # 获取评估 schema
        evaluation_schema = get_schema_for_prompt_type("evaluation")
        
        # 在评估部分
        logger.info(f"评估翻译质量...")
        try:
            evaluation_response = await self.evaluator.generate(
                prompt=EVALUATION_PROMPT_TEMPLATE.format(
                    original_text=text,
                    translated_text=translation,
                    target_language=target_language_for_prompt
                ),
                schema=evaluation_schema,
                system_prompt=self.system_prompt
            )
            
            # 解析评估结果
            evaluation_content = evaluation_response.content
            
            # 确保评估内容是正确的格式
            if isinstance(evaluation_content, dict):
                scores = evaluation_content.get("scores", {})
                total_score = evaluation_content.get("total_score", 0.0)
                feedback = evaluation_content.get("feedback", "")
                
                # 创建评估结果
                evaluation_result = EvaluationResult(
                    scores=EvaluationScores(
                        language_accuracy=float(scores.get("language_accuracy", 0.0)),
                        semantic_accuracy=float(scores.get("semantic_accuracy", 0.0)),
                        fluency=float(scores.get("fluency", 0.0)),
                        style=float(scores.get("style", 0.0)),
                        terminology=float(scores.get("terminology", 0.0))
                    ),
                    total_score=float(total_score),
                    feedback=feedback,
                    run_details=ModelRunDetails(
                        model_name=evaluation_response.model_name,
                        provider=self.evaluator_provider,
                        temperature=self.temperature,
                        prompt=EVALUATION_PROMPT_TEMPLATE.format(
                            original_text=text,
                            translated_text=translation,
                            target_language=target_language_for_prompt
                        ),
                        system_prompt=self.system_prompt,
                        input_tokens=evaluation_response.usage.get("input_tokens", 0),
                        output_tokens=evaluation_response.usage.get("output_tokens", 0),
                        total_tokens=evaluation_response.total_tokens,
                        cost=evaluation_response.cost,
                        processing_time_ms=evaluation_response.processing_time,
                        timestamp=evaluation_response.timestamp,
                        response=evaluation_content
                    )
                )
            else:
                # 如果评估内容不是字典，创建默认评估结果
                print("警告: 评估返回的内容格式不正确，使用默认评分")
                evaluation_result = EvaluationResult(
                    scores=EvaluationScores(
                        language_accuracy=5.0,
                        semantic_accuracy=5.0,
                        fluency=5.0,
                        style=5.0,
                        terminology=5.0
                    ),
                    total_score=5.0,
                    feedback="评估返回的内容格式不正确",
                    run_details=ModelRunDetails(
                        model_name=evaluation_response.model_name,
                        provider=self.evaluator_provider,
                        temperature=self.temperature,
                        prompt="评估提示",
                        system_prompt=self.system_prompt,
                        input_tokens=evaluation_response.usage.get("input_tokens", 0),
                        output_tokens=evaluation_response.usage.get("output_tokens", 0),
                        total_tokens=evaluation_response.total_tokens,
                        cost=evaluation_response.cost,
                        processing_time_ms=evaluation_response.processing_time,
                        timestamp=evaluation_response.timestamp,
                        response=str(evaluation_content)
                    )
                )
        except Exception as e:
            print(f"评估过程中出错: {str(e)}")
            # 创建默认评估结果
            evaluation_result = EvaluationResult(
                scores=EvaluationScores(
                    language_accuracy=5.0,
                    semantic_accuracy=5.0,
                    fluency=5.0,
                    style=5.0,
                    terminology=5.0
                ),
                total_score=5.0,
                feedback=f"评估过程中出错: {str(e)}",
                run_details=ModelRunDetails(
                    model_name="error",
                    provider=self.evaluator_provider,
                    temperature=self.temperature,
                    prompt="评估提示",
                    system_prompt=self.system_prompt,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    cost=0.0,
                    processing_time_ms=0,
                    timestamp=datetime.now().isoformat(),
                    response=str(e)
                )
            )
        
        # 添加评估结果到第一次尝试
        first_attempt.evaluation = evaluation_result
        
        # 添加第一次尝试到结果
        result.attempts_details.append(first_attempt)
        
        # 更新结果
        result.total_tokens += evaluation_result.run_details.total_tokens
        result.total_cost += evaluation_result.run_details.cost
        
        # 如果评分足够高，直接返回结果
        if evaluation_result.total_score >= self.min_acceptable_score:
            print(f"翻译质量良好 (评分: {evaluation_result.total_score:.1f}/10)，无需重试。")
            result.final_translation = translation
            result.final_score = evaluation_result.total_score
            result.total_time_ms = (time.perf_counter() - start_time) * 1000
            return result
        
        # 准备重试翻译
        retry_translation_schema = get_schema_for_prompt_type("retry_translation")
        
        # 重试翻译循环
        attempt_num = 1  # 已经完成了第一次尝试，从1开始
        current_translation = translation
        current_score = evaluation_result.total_score
        current_evaluation = evaluation_result  # 确保这是一个 EvaluationResult 对象
        
        while attempt_num < self.max_attempts and current_score < self.min_acceptable_score:
            logger.info(f"翻译质量不足 (评分: {current_score:.1f})，尝试第 {attempt_num + 1} 次翻译...")
            print(f"翻译质量不足 (评分: {current_score:.1f})，尝试第 {attempt_num + 1} 次翻译...")
            attempt_num += 1
            
            # 创建一个安全的评估对象用于重试翻译
            safe_scores = type('Scores', (), {
                'language_accuracy': 5.0,
                'semantic_accuracy': 5.0,
                'fluency': 5.0,
                'style': 5.0,
                'terminology': 5.0
            })
            
            safe_feedback = FeedbackObject(
                total_score=5.0,
                scores=safe_scores,
                feedback="需要改进翻译质量"
            )
            
            # 如果当前评估是有效对象，则使用其值
            if isinstance(current_evaluation, EvaluationResult):
                safe_feedback.total_score = current_evaluation.total_score
                
                if hasattr(current_evaluation, 'scores') and current_evaluation.scores:
                    safe_scores.language_accuracy = current_evaluation.scores.language_accuracy
                    safe_scores.semantic_accuracy = current_evaluation.scores.semantic_accuracy
                    safe_scores.fluency = current_evaluation.scores.fluency
                    safe_scores.style = current_evaluation.scores.style
                    safe_scores.terminology = current_evaluation.scores.terminology
                
                if hasattr(current_evaluation, 'feedback') and current_evaluation.feedback:
                    safe_feedback.feedback = current_evaluation.feedback
            
            try:
                # 重试翻译
                retry_prompt = RETRY_TRANSLATION_PROMPT_TEMPLATE.format(
                    text=text,
                    target_language=target_language_for_prompt,
                    previous_translation=current_translation,
                    feedback=safe_feedback
                )
                
                retry_response = await self.retry_translator.generate(
                    prompt=retry_prompt,
                    schema=retry_translation_schema,
                    system_prompt=self.system_prompt
                )
                
                logger.debug(f"重试翻译响应: {retry_response}")
                
                # 解析重试翻译结果
                retry_content = retry_response.content
                logger.debug(f"重试翻译内容: {retry_content}")
                
                # 确保 retry_content 是字典类型
                if isinstance(retry_content, dict):
                    retry_translation = retry_content.get("translated_text", current_translation)
                elif isinstance(retry_content, list) and len(retry_content) > 0:
                    # 如果是列表，尝试获取第一个元素
                    if isinstance(retry_content[0], dict):
                        retry_translation = retry_content[0].get("translated_text", current_translation)
                    else:
                        retry_translation = str(retry_content[0])
                elif isinstance(retry_content, str):
                    # 如果是字符串，直接使用
                    retry_translation = retry_content
                else:
                    # 其他情况，使用当前翻译
                    logger.warning(f"警告: 重试翻译返回了意外的格式: {type(retry_content)}")
                    retry_translation = current_translation
            
            except Exception as e:
                logger.error(f"重试翻译过程中出错: {str(e)}", exc_info=True)
                # 使用当前翻译作为重试翻译结果
                retry_translation = current_translation
                
                # 创建一个默认的响应对象
                retry_content = {"translated_text": current_translation}
                retry_response = LLMResponse(
                    content=retry_content,
                    model_name="error",
                    usage={"input_tokens": 0, "output_tokens": 0},
                    total_tokens=0,
                    cost=0.0,
                    processing_time=0,
                    timestamp=datetime.now().isoformat()
                )
            
            # 更新当前翻译
            current_translation = retry_translation
            
            # 记录重试翻译详情
            retry_attempt = TranslationAttempt(
                attempt=attempt_num,
                translation=retry_translation,
                source_language="unknown",  # 重试翻译不需要再次检测源语言
                source_language_code="und",
                source_language_name="未知语言",
                source_language_autonym="未知语言",
                explanation="重试翻译",
                run_details=ModelRunDetails(
                    model_name=retry_response.model_name,
                    provider=self.retry_translator_provider,
                    temperature=self.temperature,
                    prompt="重试翻译提示",
                    system_prompt=self.system_prompt,
                    input_tokens=retry_response.usage.get("input_tokens", 0),
                    output_tokens=retry_response.usage.get("output_tokens", 0),
                    total_tokens=retry_response.total_tokens,
                    cost=retry_response.cost,
                    processing_time_ms=retry_response.processing_time,
                    timestamp=retry_response.timestamp,
                    response=retry_content
                )
            )
            
            # 更新结果
            result.attempts += 1
            result.total_tokens += retry_response.total_tokens
            result.total_cost += retry_response.cost
            
            # 评估重试翻译质量
            logger.info(f"评估重试翻译质量...")
            try:
                retry_evaluation_response = await self.evaluator.generate(
                    prompt=EVALUATION_PROMPT_TEMPLATE.format(
                        original_text=text,
                        translated_text=retry_translation,
                        target_language=target_language_for_prompt
                    ),
                    schema=get_schema_for_prompt_type("evaluation"),
                    system_prompt=self.system_prompt
                )
                
                logger.debug(f"重试评估响应: {retry_evaluation_response}")
                
                # 解析评估结果
                retry_evaluation_content = retry_evaluation_response.content
                
                # 在评估结果处理部分
                try:
                    # 解析评估内容
                    if isinstance(retry_evaluation_content, str):
                        # 如果是字符串，尝试解析为字典
                        try:
                            import ast
                            retry_evaluation_content_dict = ast.literal_eval(retry_evaluation_content)
                            if isinstance(retry_evaluation_content_dict, dict):
                                retry_evaluation_content = retry_evaluation_content_dict
                            else:
                                retry_evaluation_content = {"content": retry_evaluation_content}
                        except:
                            retry_evaluation_content = {"content": retry_evaluation_content}
                    
                    # 确保 response 字段是字典
                    response_dict = retry_evaluation_content
                    if not isinstance(response_dict, dict):
                        response_dict = {"content": str(retry_evaluation_content)}
                    
                    # 创建 ModelRunDetails
                    run_details = ModelRunDetails(
                        model_name=retry_evaluation_response.model_name,
                        provider=self.evaluator_provider,
                        temperature=self.temperature,
                        prompt="评估提示",
                        system_prompt=self.system_prompt,
                        input_tokens=retry_evaluation_response.usage.get("input_tokens", 0),
                        output_tokens=retry_evaluation_response.usage.get("output_tokens", 0),
                        total_tokens=retry_evaluation_response.total_tokens,
                        cost=retry_evaluation_response.cost,
                        processing_time_ms=retry_evaluation_response.processing_time,
                        timestamp=retry_evaluation_response.timestamp,
                        response=response_dict  # 使用确保是字典的 response_dict
                    )
                except Exception as e:
                    logger.error(f"处理评估响应时出错: {str(e)}", exc_info=True)
                    # 创建默认的 response 字典
                    response_dict = {"error": str(e)}
                    
                    # 创建默认的 ModelRunDetails
                    run_details = ModelRunDetails(
                        model_name="error",
                        provider=self.evaluator_provider,
                        temperature=self.temperature,
                        prompt="评估提示",
                        system_prompt=self.system_prompt,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        cost=0.0,
                        processing_time_ms=0,
                        timestamp=datetime.now().isoformat(),
                        response=response_dict
                    )
                
                # 创建评估结果
                retry_evaluation_result = EvaluationResult(
                    scores=EvaluationScores(
                        language_accuracy=float(run_details.response.get("language_accuracy", 0.0)),
                        semantic_accuracy=float(run_details.response.get("semantic_accuracy", 0.0)),
                        fluency=float(run_details.response.get("fluency", 0.0)),
                        style=float(run_details.response.get("style", 0.0)),
                        terminology=float(run_details.response.get("terminology", 0.0))
                    ),
                    total_score=float(run_details.response.get("total_score", 0.0)),
                    feedback=run_details.response.get("feedback", ""),
                    run_details=run_details
                )
            except Exception as e:
                logger.error(f"评估过程中出错: {str(e)}", exc_info=True)
                # 创建默认评估结果和响应对象
                retry_evaluation_response = LLMResponse(
                    content={"scores": {}, "total_score": 5.0, "feedback": f"评估过程中出错: {str(e)}"},
                    model_name="error",
                    usage={"input_tokens": 0, "output_tokens": 0},
                    total_tokens=0,
                    cost=0.0,
                    processing_time=0,
                    timestamp=datetime.now().isoformat()
                )
                
                retry_evaluation_result = EvaluationResult(
                    scores=EvaluationScores(
                        language_accuracy=5.0,
                        semantic_accuracy=5.0,
                        fluency=5.0,
                        style=5.0,
                        terminology=5.0
                    ),
                    total_score=5.0,
                    feedback=f"评估过程中出错: {str(e)}",
                    run_details=ModelRunDetails(
                        model_name="error",
                        provider=self.evaluator_provider,
                        temperature=self.temperature,
                        prompt="评估提示",
                        system_prompt=self.system_prompt,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        cost=0.0,
                        processing_time_ms=0,
                        timestamp=datetime.now().isoformat(),
                        response={"error": str(e)}  # 确保这是一个字典
                    )
                )
            
            # 添加评估结果到重试尝试
            retry_attempt.evaluation = retry_evaluation_result
            
            # 添加重试尝试到结果
            result.attempts_details.append(retry_attempt)
            
            # 更新结果
            result.total_tokens += retry_evaluation_result.run_details.total_tokens
            result.total_cost += retry_evaluation_result.run_details.cost
            
            # 更新当前评分和评估结果
            current_score = retry_evaluation_result.total_score
            current_evaluation = retry_evaluation_result
            
            print(f"重试翻译评分: {current_score:.1f}/10")
        
        # 在最终结果部分
        logger.info(f"翻译完成: 最终评分={current_score:.1f}/10, 尝试次数={attempt_num}")
        logger.debug(f"最终翻译结果: {current_translation[:50]}...")
        
        # 设置最终结果
        result.final_translation = current_translation
        result.final_score = current_score
        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result

# 同步包装类，用于命令行调用
class TranslationPipeline:
    def __init__(self, **kwargs):
        self.async_pipeline = AsyncTranslationPipeline(**kwargs)
    
    def translate(self, text: str, target_language: str) -> TranslationResult:
        """同步翻译方法"""
        return asyncio.run(self.async_pipeline.translate(text, target_language))

# 主函数，用于测试
async def main():
    # 加载环境变量
    load_dotenv()
    
    # 创建翻译 pipeline
    pipeline = AsyncTranslationPipeline(
        translator_provider="openai",
        evaluator_provider="anthropic",
        retry_translator_provider="google",
        temperature=0.1,
        max_tokens=1000,
        max_attempts=3,
        min_acceptable_score=8.0,
        display_language="zh"  # 使用中文显示语言名称
    )
    
    # 测试文本
    text = "Artificial intelligence is transforming the way we live and work. It offers tremendous opportunities but also presents significant challenges that we must address."
    target_language = "中文"  # 可以是语言名称或代码，例如 "zh" 或 "中文"
    
    # 翻译文本
    result = await pipeline.translate(text, target_language)
    
    # 打印结果
    print("\n" + "=" * 50)
    print(f"原文: {result.original_text}")
    print(f"目标语言: {result.target_language} ({result.target_language_code})")
    print(f"目标语言名称: {result.target_language_name} / {result.target_language_autonym}")
    print(f"源语言: {result.source_language} ({result.source_language_code})")
    print(f"源语言名称: {result.source_language_name} / {result.source_language_autonym}")
    print(f"最终翻译: {result.final_translation}")
    print(f"最终评分: {result.final_score:.1f}/10")
    print(f"尝试次数: {result.attempts}")
    
    # 计算每个步骤的统计信息
    translation_tokens = 0
    translation_cost = 0.0
    translation_time = 0.0
    
    evaluation_tokens = 0
    evaluation_cost = 0.0
    evaluation_time = 0.0
    
    retry_tokens = 0
    retry_cost = 0.0
    retry_time = 0.0
    
    # 遍历所有尝试，计算统计信息
    for attempt in result.attempts_details:
        if attempt.attempt == 1:  # 初始翻译
            translation_tokens += attempt.run_details.total_tokens
            translation_cost += attempt.run_details.cost
            translation_time += attempt.run_details.processing_time_ms
            
            if attempt.evaluation:  # 初始评估
                evaluation_tokens += attempt.evaluation.run_details.total_tokens
                evaluation_cost += attempt.evaluation.run_details.cost
                evaluation_time += attempt.evaluation.run_details.processing_time_ms
        else:  # 重试翻译
            retry_tokens += attempt.run_details.total_tokens
            retry_cost += attempt.run_details.cost
            retry_time += attempt.run_details.processing_time_ms
            
            if attempt.evaluation:  # 重试评估
                evaluation_tokens += attempt.evaluation.run_details.total_tokens
                evaluation_cost += attempt.evaluation.run_details.cost
                evaluation_time += attempt.evaluation.run_details.processing_time_ms
    
    # 打印详细统计信息
    print("\n统计信息:")
    print(f"初始翻译: {translation_tokens} Token, ${translation_cost:.6f}, {translation_time/1000:.2f}秒")
    print(f"评估: {evaluation_tokens} Token, ${evaluation_cost:.6f}, {evaluation_time/1000:.2f}秒")
    if result.attempts > 1:
        print(f"重试翻译: {retry_tokens} Token, ${retry_cost:.6f}, {retry_time/1000:.2f}秒")
    print(f"总计: {result.total_tokens} Token, ${result.total_cost:.6f}, {result.total_time_ms/1000:.2f}秒")
    
    # 打印每次尝试的详情
    print("\n尝试详情:")
    for i, attempt in enumerate(result.attempts_details):
        print(f"\n尝试 {i+1}:")
        print(f"  源语言: {attempt.source_language} ({attempt.source_language_code})")
        print(f"  翻译: {attempt.translation}")
        if attempt.evaluation:
            print(f"  评分: {attempt.evaluation.total_score:.1f}/10")
            print(f"  语言准确性: {attempt.evaluation.scores.language_accuracy:.1f}")
            print(f"  语义准确性: {attempt.evaluation.scores.semantic_accuracy:.1f}")
            print(f"  流畅性: {attempt.evaluation.scores.fluency:.1f}")
            print(f"  风格: {attempt.evaluation.scores.style:.1f}")
            print(f"  术语准确性: {attempt.evaluation.scores.terminology:.1f}")
            print(f"  反馈: {attempt.evaluation.feedback}")
    
    # 保存详细结果到JSON文件
    with open("translation_result.json", "w", encoding="utf-8") as f:
        json.dump(result.dict(), f, ensure_ascii=False, indent=2, default=str)
    
    print("-" * 50)
    print("详细结果已保存到 translation_result.json")

if __name__ == "__main__":
    asyncio.run(main())