#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
翻译测试脚本

此脚本用于测试翻译管道的功能，包括多种语言和模型组合。
从外部 JSON 文件加载测试文本和模型组合。
"""

import asyncio
import json
import os
import csv
import pandas as pd
from dotenv import load_dotenv
from translate_pipeline import AsyncTranslationPipeline

async def test_translation():
    # 加载环境变量
    load_dotenv()
    
    # 加载测试文本
    try:
        with open("test_texts.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
            # 检查是否有嵌套结构
            if isinstance(test_data, dict) and "texts" in test_data:
                TEST_TEXTS = test_data["texts"]
            elif isinstance(test_data, list):
                # 如果是直接的列表，检查每个元素是否是字典
                if all(isinstance(item, dict) for item in test_data):
                    TEST_TEXTS = test_data
                else:
                    # 如果是字符串列表，转换为所需格式
                    TEST_TEXTS = [{"text": item, "target_language": "中文", "description": f"测试 {i+1}"} 
                                 for i, item in enumerate(test_data)]
            else:
                print("错误: test_texts.json 格式不符合预期")
                return
        print(f"已加载 {len(TEST_TEXTS)} 个测试文本")
    except FileNotFoundError:
        print("错误: 未找到 test_texts.json 文件")
        return
    except json.JSONDecodeError:
        print("错误: test_texts.json 文件格式不正确")
        return
    
    # 加载模型组合
    try:
        with open("model_combinations.json", "r", encoding="utf-8") as f:
            model_data = json.load(f)
            MODEL_COMBINATIONS = model_data.get("combinations", [])
        print(f"已加载 {len(MODEL_COMBINATIONS)} 个模型组合")
    except FileNotFoundError:
        print("错误: 未找到 model_combinations.json 文件")
        return
    except json.JSONDecodeError:
        print("错误: model_combinations.json 文件格式不正确")
        return
    
    # 创建结果目录
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建结果列表
    all_results = []
    
    # 创建CSV数据列表
    csv_data = []
    
    # 测试每个模型组合
    for model_combo in MODEL_COMBINATIONS:
        print(f"\n\n===== 测试模型组合: {model_combo['name']} =====")
        
        # 创建翻译管道
        pipeline = AsyncTranslationPipeline(
            translator_provider=model_combo["translator"][0],
            translator_model=model_combo["translator"][1],
            evaluator_provider=model_combo["evaluator"][0],
            evaluator_model=model_combo["evaluator"][1],
            retry_translator_provider=model_combo["retry_translator"][0],
            retry_translator_model=model_combo["retry_translator"][1],
            temperature=0.1,
            max_tokens=2000,
            max_attempts=2,
            min_acceptable_score=7.0,
            display_language="zh"
        )
        
        # 测试结果
        model_results = {
            "model_combination": model_combo,
            "test_results": []
        }
        
        # 测试每个文本
        for test in TEST_TEXTS:
            print(f"\n测试: {test['description']}")
            print(f"原文: {test['text']}")
            print(f"目标语言: {test['target_language']}")
            
            try:
                # 翻译文本
                result = await pipeline.translate(test['text'], test['target_language'])
                
                # 打印结果
                print(f"源语言: {result.source_language} ({result.source_language_code})")
                print(f"翻译: {result.final_translation}")
                print(f"评分: {result.final_score:.1f}/10")
                print(f"尝试次数: {result.attempts}")
                
                # 添加到结果列表
                model_results["test_results"].append({
                    "test": test,
                    "result": result.model_dump()  # 使用 model_dump 替代 dict
                })
                
                # 添加到CSV数据
                csv_row = {
                    "模型组合": model_combo["name"],
                    "测试描述": test["description"],
                    "原文": test["text"],
                    "目标语言": test["target_language"],
                    "源语言": f"{result.source_language} ({result.source_language_code})",
                    "翻译结果": result.final_translation,
                    "评分": f"{result.final_score:.1f}/10",
                    "尝试次数": result.attempts,
                    "总Token": result.total_tokens,
                    "总成本($)": f"${result.total_cost:.8f}",  # 美元，8位小数
                    "总成本(¢)": f"{result.total_cost * 100:.6f}¢",  # 美分，6位小数
                    "总时间(秒)": f"{result.total_time_ms/1000:.2f}"
                }
                
                # 如果有多次尝试，添加每次尝试的详情
                if result.attempts > 1:
                    for i, attempt in enumerate(result.attempts_details):
                        csv_row[f"尝试{i+1}翻译"] = attempt.translation
                        if attempt.evaluation:
                            csv_row[f"尝试{i+1}评分"] = f"{attempt.evaluation.total_score:.1f}/10"
                
                csv_data.append(csv_row)
                
            except Exception as e:
                print(f"测试失败: {str(e)}")
                # 添加错误信息到结果列表
                model_results["test_results"].append({
                    "test": test,
                    "error": str(e)
                })
                
                # 添加错误信息到CSV数据
                csv_data.append({
                    "模型组合": model_combo["name"],
                    "测试描述": test["description"],
                    "原文": test["text"],
                    "目标语言": test["target_language"],
                    "错误": str(e)
                })
        
        # 添加到总结果
        all_results.append(model_results)
        
        # 为每个模型组合保存单独的结果文件
        combo_name = model_combo["name"].replace(" ", "_").lower()
        combo_result_file = os.path.join(results_dir, f"{combo_name}_results.json")
        with open(combo_result_file, "w", encoding="utf-8") as f:
            json.dump(model_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"模型组合 {model_combo['name']} 的结果已保存到 {combo_result_file}")
    
    # 保存所有结果到JSON文件
    all_results_file = os.path.join(results_dir, "all_translation_test_results.json")
    with open(all_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n所有测试结果已保存到 {all_results_file}")
    
    # 保存CSV结果
    # 方法1: 使用pandas
    try:
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(results_dir, "translation_test_results.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')  # 使用带BOM的UTF-8编码，Excel可以正确识别中文
        print(f"CSV结果已保存到 {csv_file}")
    except Exception as e:
        print(f"保存CSV时出错 (pandas): {str(e)}")
        
        # 方法2: 使用csv模块作为备选
        try:
            csv_file = os.path.join(results_dir, "translation_test_results_backup.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
                    print(f"备用CSV结果已保存到 {csv_file}")
        except Exception as e2:
            print(f"保存CSV时出错 (csv模块): {str(e2)}")
    
    # 创建一个简化版的CSV，只包含关键信息
    try:
        simple_csv_data = []
        for row in csv_data:
            if "错误" in row:
                simple_row = {
                    "模型组合": row["模型组合"],
                    "测试描述": row["测试描述"],
                    "原文": row["原文"],
                    "目标语言": row["目标语言"],
                    "错误": row["错误"]
                }
            else:
                simple_row = {
                    "模型组合": row["模型组合"],
                    "测试描述": row["测试描述"],
                    "原文": row["原文"],
                    "目标语言": row["目标语言"],
                    "源语言": row["源语言"],
                    "翻译结果": row["翻译结果"],
                    "评分": row["评分"],
                    "尝试次数": row["尝试次数"]
                }
            simple_csv_data.append(simple_row)
            
        simple_df = pd.DataFrame(simple_csv_data)
        simple_csv_file = os.path.join(results_dir, "translation_test_results_simple.csv")
        simple_df.to_csv(simple_csv_file, index=False, encoding='utf-8-sig')
        print(f"简化版CSV结果已保存到 {simple_csv_file}")
    except Exception as e:
        print(f"保存简化版CSV时出错: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_translation()) 