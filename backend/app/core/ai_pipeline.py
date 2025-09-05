"""
AI对话处理流水线

实现完整的AI对话处理流程，包括输入预处理、上下文管理、记忆检索、
人设应用、AI生成和后处理等步骤
"""

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from uuid import UUID

import openai
from openai import AsyncOpenAI

from ..config import get_settings
from ..models.chat_models import ChatMessage, MessageRole, MessageCreate, ConversationContext
from ..models.memory_models import Memory
from .context_manager import ContextManager
from .memory_manager import MemoryManager
from .character_system import CharacterSystem
from .stream_handler import StreamHandler
from ..services.ollama_service import get_ollama_service
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AIPipeline:
    """AI对话处理流水线"""
    
    def __init__(self):
        self.context_manager = ContextManager()
        self.memory_manager = MemoryManager()
        self.character_system = CharacterSystem()
        self.stream_handler = StreamHandler()
        
        # AI客户端
        self.ollama_service = None
        self.openai_client = None
        self.anthropic_client = None
        
        self._init_ai_clients_sync()
    
    async def _init_ai_clients(self):
        """初始化AI客户端"""
        try:
            # Ollama服务
            self.ollama_service = await get_ollama_service()
            if self.ollama_service and self.ollama_service.available_models:
                logger.info(f"Ollama服务初始化成功，可用模型: {len(self.ollama_service.available_models)}个")
            else:
                logger.warning("Ollama服务初始化失败或无可用模型")
            
            # OpenAI客户端
            if settings.openai_api_key:
                self.openai_client = AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI客户端初始化成功")
            
            # Anthropic客户端（如果需要）
            # if settings.anthropic_api_key:
            #     import anthropic
            #     self.anthropic_client = anthropic.AsyncAnthropic(
            #         api_key=settings.anthropic_api_key
            #     )
            
        except Exception as e:
            logger.error(f"初始化AI客户端失败: {e}")
    
    def _init_ai_clients_sync(self):
        """同步初始化AI客户端（用于__init__）"""
        try:
            # OpenAI客户端
            if settings.openai_api_key:
                self.openai_client = AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI客户端初始化成功")
            
        except Exception as e:
            logger.error(f"初始化AI客户端失败: {e}")
    
    async def process_message(
        self,
        user_id: UUID,
        session_id: UUID,
        message: MessageCreate,
        character_id: str = "default",
        stream_id: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """处理用户消息的完整流水线"""
        
        start_time = time.time()
        processing_metadata = {
            "start_time": start_time,
            "steps_completed": [],
            "tokens_used": 0,
            "model_used": None,
            "character_used": character_id,
            "memory_retrieved": [],
            "processing_time": 0
        }
        
        try:
            # 第1步：输入预处理与分析
            logger.info(f"开始处理消息: 用户{user_id}, 会话{session_id}")
            
            processed_input = await self._preprocess_input(message.content)
            processing_metadata["steps_completed"].append("input_preprocessing")
            
            # 第2步：意图识别与实体提取
            intent_analysis = await self._analyze_intent(processed_input)
            processing_metadata["intent"] = intent_analysis
            processing_metadata["steps_completed"].append("intent_analysis")
            
            # 第3步：获取历史消息和上下文
            # 这里应该从数据库获取，暂时使用空列表
            historical_messages = []  # await self._get_historical_messages(session_id)
            
            # 第4步：长期记忆检索
            relevant_memories = await self.memory_manager.retrieve_relevant_memories(
                user_id=user_id,
                query=processed_input,
                limit=settings.memory.retrieval_max_results
            )
            
            memory_context = self._build_memory_context(relevant_memories)
            processing_metadata["memory_retrieved"] = [str(mem[0].id) for mem in relevant_memories]
            processing_metadata["steps_completed"].append("memory_retrieval")
            
            # 第5步：人设应用
            character_config = self.character_system.get_character(character_id)
            processing_metadata["steps_completed"].append("character_loading")
            
            # 第6步：上下文构建与优化
            context = await self.context_manager.build_context(
                session_id=session_id,
                user_input=processed_input,
                messages=historical_messages,
                character_config=character_config,
                memory_context=memory_context
            )
            
            processing_metadata["tokens_used"] = context.total_tokens
            processing_metadata["steps_completed"].append("context_building")
            
            # 第7步：构建AI提示
            formatted_messages = self.context_manager.format_context_for_ai(
                context=context,
                character_config=character_config,
                memory_context=memory_context,
                user_input=processed_input
            )
            
            processing_metadata["steps_completed"].append("prompt_engineering")
            
            # 第8步：AI生成
            if stream_id and self.stream_handler.is_stream_active(stream_id):
                # 流式生成
                response = await self._generate_streaming_response(
                    formatted_messages, character_config, stream_id, processing_metadata
                )
            else:
                # 非流式生成
                response = await self._generate_response(
                    formatted_messages, character_config, processing_metadata
                )
            
            processing_metadata["steps_completed"].append("ai_generation")
            
            # 第9步：人设一致性检查和调整
            adjusted_response, was_adjusted = await self.character_system.apply_character_consistency(
                response, character_config, historical_messages
            )
            
            if was_adjusted:
                processing_metadata["response_adjusted"] = True
                response = adjusted_response
            
            processing_metadata["steps_completed"].append("consistency_check")
            
            # 第10步：后处理与优化
            final_response = await self._postprocess_response(
                response, character_config, intent_analysis
            )
            processing_metadata["steps_completed"].append("postprocessing")
            
            # 第11步：记忆更新
            await self._update_memories(
                user_id, session_id, processed_input, final_response, context
            )
            processing_metadata["steps_completed"].append("memory_update")
            
            # 计算总处理时间
            processing_metadata["processing_time"] = time.time() - start_time
            
            logger.info(
                f"消息处理完成: {len(processing_metadata['steps_completed'])}个步骤, "
                f"耗时{processing_metadata['processing_time']:.2f}秒"
            )
            
            return final_response, processing_metadata
            
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            processing_metadata["error"] = str(e)
            processing_metadata["processing_time"] = time.time() - start_time
            
            # 返回错误处理响应
            error_response = await self._generate_error_response(str(e), character_config)
            return error_response, processing_metadata
    
    async def _preprocess_input(self, user_input: str) -> str:
        """输入预处理与标准化"""
        try:
            # 1. 基础清理
            processed = user_input.strip()
            
            # 2. 去除多余空白字符
            processed = re.sub(r'\s+', ' ', processed)
            
            # 3. 标准化标点符号
            punctuation_map = {
                '！': '!',
                '？': '?',
                '，': ',',
                '。': '.',
                '；': ';',
                '：': ':',
                '"': '"',
                '"': '"',
                ''': "'",
                ''': "'"
            }
            
            for chinese_punct, english_punct in punctuation_map.items():
                processed = processed.replace(chinese_punct, english_punct)
            
            # 4. 处理特殊字符
            processed = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:\'\"()[\]{}@#$%^&*+=<>/-]', '', processed)
            
            # 5. 长度检查
            if len(processed) > 2000:
                processed = processed[:2000]
                logger.warning("输入内容过长，已截断")
            
            return processed
            
        except Exception as e:
            logger.error(f"输入预处理失败: {e}")
            return user_input
    
    async def _analyze_intent(self, processed_input: str) -> Dict[str, Any]:
        """意图识别与实体提取"""
        try:
            intent_analysis = {
                "primary_intent": "unknown",
                "confidence": 0.0,
                "entities": [],
                "sentiment": "neutral",
                "urgency": "normal",
                "topic_category": "general"
            }
            
            input_lower = processed_input.lower()
            
            # 简单的意图识别
            intent_patterns = {
                "question": ["?", "？", "什么", "怎么", "如何", "为什么", "哪里", "谁", "when", "what", "how", "why"],
                "request": ["请", "帮我", "能否", "可以", "需要", "想要", "希望"],
                "greeting": ["你好", "hi", "hello", "早上好", "下午好", "晚上好"],
                "farewell": ["再见", "拜拜", "goodbye", "bye", "结束"],
                "complaint": ["问题", "错误", "不对", "不行", "故障", "bug"],
                "praise": ["好", "棒", "不错", "优秀", "perfect", "excellent"],
                "casual_chat": ["聊天", "无聊", "随便", "闲聊"]
            }
            
            max_confidence = 0.0
            detected_intent = "unknown"
            
            for intent, keywords in intent_patterns.items():
                matches = sum(1 for keyword in keywords if keyword in input_lower)
                confidence = matches / len(keywords)
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_intent = intent
            
            intent_analysis["primary_intent"] = detected_intent
            intent_analysis["confidence"] = min(max_confidence * 2, 1.0)  # 调整置信度
            
            # 情感分析
            positive_words = ["好", "棒", "喜欢", "满意", "开心", "高兴", "谢谢"]
            negative_words = ["不好", "糟糕", "讨厌", "失望", "难过", "生气", "问题"]
            
            positive_count = sum(1 for word in positive_words if word in input_lower)
            negative_count = sum(1 for word in negative_words if word in input_lower)
            
            if positive_count > negative_count:
                intent_analysis["sentiment"] = "positive"
            elif negative_count > positive_count:
                intent_analysis["sentiment"] = "negative"
            
            # 紧急程度判断
            urgent_keywords = ["紧急", "急", "马上", "立即", "快", "urgent", "asap"]
            if any(keyword in input_lower for keyword in urgent_keywords):
                intent_analysis["urgency"] = "high"
            
            # 主题分类
            topic_keywords = {
                "technical": ["代码", "编程", "bug", "系统", "软件", "技术"],
                "business": ["工作", "项目", "会议", "客户", "业务"],
                "personal": ["生活", "家庭", "朋友", "感情", "健康"],
                "learning": ["学习", "课程", "知识", "教程", "学会"]
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in input_lower for keyword in keywords):
                    intent_analysis["topic_category"] = topic
                    break
            
            return intent_analysis
            
        except Exception as e:
            logger.error(f"意图分析失败: {e}")
            return {"primary_intent": "unknown", "confidence": 0.0}
    
    def _build_memory_context(self, relevant_memories: List[Tuple[Memory, float]]) -> str:
        """构建记忆上下文"""
        if not relevant_memories:
            return ""
        
        try:
            memory_parts = []
            
            for memory, similarity in relevant_memories[:5]:  # 只使用最相关的5条记忆
                memory_text = f"{memory.content} (相关度: {similarity:.2f})"
                memory_parts.append(memory_text)
            
            return "相关记忆: " + "; ".join(memory_parts)
            
        except Exception as e:
            logger.error(f"构建记忆上下文失败: {e}")
            return ""
    
    async def _generate_response(
        self,
        messages: List[Dict[str, str]],
        character_config: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """生成AI响应（非流式）"""
        try:
            # 确保Ollama服务已初始化
            if not self.ollama_service:
                await self._init_ai_clients()
            
            # 配置生成参数
            model_config = settings.ai.model
            provider = model_config.provider
            
            # 根据提供商选择不同的生成方法
            if provider == "ollama" and self.ollama_service:
                # 使用Ollama生成
                model_name = model_config.name
                
                # 如果指定的模型不可用，尝试推荐一个
                if not self.ollama_service.is_model_available(model_name):
                    recommended_model = await self.ollama_service.suggest_model_for_task("chat")
                    if recommended_model:
                        model_name = recommended_model
                        logger.info(f"使用推荐模型: {model_name}")
                    else:
                        return "抱歉，没有可用的Ollama模型。"
                
                generated_text = await self.ollama_service.generate_response(
                    model=model_name,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    top_p=model_config.top_p
                )
                
                # 更新元数据
                metadata["model_used"] = model_name
                metadata["provider"] = "ollama"
                metadata["tokens_used"] += len(generated_text.split())  # 估算token数
                
                return generated_text or "抱歉，我暂时无法回复。"
                
            elif provider == "openai" and self.openai_client:
                # 使用OpenAI生成
                response = await self.openai_client.chat.completions.create(
                    model=model_config.name,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    top_p=model_config.top_p,
                    frequency_penalty=model_config.frequency_penalty,
                    presence_penalty=model_config.presence_penalty
                )
                
                generated_text = response.choices[0].message.content
                
                # 更新元数据
                metadata["model_used"] = model_config.name
                metadata["provider"] = "openai"
                metadata["tokens_used"] += response.usage.total_tokens
                
                return generated_text or "抱歉，我暂时无法回复。"
            
            else:
                # 回退到默认错误信息
                available_providers = []
                if self.ollama_service and self.ollama_service.available_models:
                    available_providers.append("ollama")
                if self.openai_client:
                    available_providers.append("openai")
                
                if available_providers:
                    return f"抱歉，配置的AI提供商 '{provider}' 不可用。可用的提供商: {', '.join(available_providers)}"
                else:
                    return "抱歉，AI服务暂时不可用。"
            
        except Exception as e:
            logger.error(f"生成AI响应失败: {e}")
            return f"抱歉，处理您的请求时出现了问题: {str(e)}"
    
    async def _generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        character_config: Dict[str, Any],
        stream_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """生成流式AI响应"""
        try:
            # 确保Ollama服务已初始化
            if not self.ollama_service:
                await self._init_ai_clients()
            
            # 配置生成参数
            model_config = settings.ai.model
            provider = model_config.provider
            
            # 发送开始信号
            await self.stream_handler.send_status_update(
                stream_id, "generating", "AI正在思考中..."
            )
            
            # 根据提供商选择不同的流式生成方法
            if provider == "ollama" and self.ollama_service:
                # 使用Ollama流式生成
                model_name = model_config.name
                
                # 如果指定的模型不可用，尝试推荐一个
                if not self.ollama_service.is_model_available(model_name):
                    recommended_model = await self.ollama_service.suggest_model_for_task("chat")
                    if recommended_model:
                        model_name = recommended_model
                        logger.info(f"使用推荐模型: {model_name}")
                    else:
                        error_msg = "抱歉，没有可用的Ollama模型。"
                        await self.stream_handler.send_stream_chunk(
                            stream_id, error_msg, is_final=True
                        )
                        return error_msg
                
                # 处理流式响应
                full_response = ""
                
                async def ollama_response_generator():
                    nonlocal full_response
                    async for token in self.ollama_service.generate_streaming_response(
                        model=model_name,
                        messages=messages,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens,
                        top_p=model_config.top_p
                    ):
                        full_response += token
                        yield token
                
                # 使用流处理器传输响应
                await self.stream_handler.stream_ai_response(
                    stream_id, ollama_response_generator(), character_config
                )
                
                # 更新元数据
                metadata["model_used"] = model_name
                metadata["provider"] = "ollama"
                metadata["tokens_used"] += len(full_response.split())  # 估算token数
                
                return full_response or "抱歉，我暂时无法回复。"
                
            elif provider == "openai" and self.openai_client:
                # 使用OpenAI流式生成
                # 创建流式响应
                stream = await self.openai_client.chat.completions.create(
                    model=model_config.name,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    top_p=model_config.top_p,
                    frequency_penalty=model_config.frequency_penalty,
                    presence_penalty=model_config.presence_penalty,
                    stream=True
                )
                
                # 处理流式响应
                full_response = ""
                
                async def openai_response_generator():
                    nonlocal full_response
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            full_response += token
                            yield token
                
                # 使用流处理器传输响应
                await self.stream_handler.stream_ai_response(
                    stream_id, openai_response_generator(), character_config
                )
                
                # 更新元数据
                metadata["model_used"] = model_config.name
                metadata["provider"] = "openai"
                metadata["tokens_used"] += len(full_response.split())  # 简单估算
                
                return full_response or "抱歉，我暂时无法回复。"
            
            else:
                # 回退到错误信息
                available_providers = []
                if self.ollama_service and self.ollama_service.available_models:
                    available_providers.append("ollama")
                if self.openai_client:
                    available_providers.append("openai")
                
                if available_providers:
                    error_msg = f"抱歉，配置的AI提供商 '{provider}' 不可用。可用的提供商: {', '.join(available_providers)}"
                else:
                    error_msg = "抱歉，AI服务暂时不可用。"
                
                await self.stream_handler.send_stream_chunk(
                    stream_id, error_msg, is_final=True
                )
                return error_msg
            
        except Exception as e:
            logger.error(f"生成流式AI响应失败: {e}")
            error_msg = f"抱歉，处理您的请求时出现了问题: {str(e)}"
            
            await self.stream_handler.send_stream_chunk(
                stream_id, error_msg, is_final=True
            )
            
            return error_msg
    
    async def _postprocess_response(
        self,
        response: str,
        character_config: Dict[str, Any],
        intent_analysis: Dict[str, Any]
    ) -> str:
        """后处理与优化"""
        try:
            processed_response = response
            
            # 1. 敏感内容过滤
            processed_response = await self._filter_sensitive_content(processed_response)
            
            # 2. 格式化处理
            processed_response = self._format_response(processed_response)
            
            # 3. 根据意图调整回复
            processed_response = await self._adjust_response_for_intent(
                processed_response, intent_analysis, character_config
            )
            
            # 4. 长度检查
            max_length = character_config.get("constraints", {}).get("response_length", {}).get("max", 1000)
            if len(processed_response) > max_length:
                processed_response = processed_response[:max_length-3] + "..."
            
            return processed_response
            
        except Exception as e:
            logger.error(f"后处理响应失败: {e}")
            return response
    
    async def _filter_sensitive_content(self, response: str) -> str:
        """过滤敏感内容"""
        try:
            # 简单的敏感词过滤
            sensitive_words = [
                "政治", "暴力", "色情", "赌博", "毒品",
                # 可以根据需要添加更多敏感词
            ]
            
            filtered_response = response
            for word in sensitive_words:
                if word in filtered_response:
                    filtered_response = filtered_response.replace(word, "*" * len(word))
            
            return filtered_response
            
        except Exception as e:
            logger.error(f"敏感内容过滤失败: {e}")
            return response
    
    def _format_response(self, response: str) -> str:
        """格式化回复"""
        try:
            # 1. 去除多余空白
            formatted = re.sub(r'\s+', ' ', response.strip())
            
            # 2. 确保句子结尾有标点
            if formatted and not formatted[-1] in '.!?。！？':
                formatted += '。'
            
            # 3. 修正常见的格式问题
            formatted = re.sub(r'\s+([,.!?。，！？])', r'\1', formatted)  # 标点前不应有空格
            formatted = re.sub(r'([,.!?。，！？])\s*([a-zA-Z\u4e00-\u9fff])', r'\1 \2', formatted)  # 标点后应有空格
            
            return formatted
            
        except Exception as e:
            logger.error(f"格式化回复失败: {e}")
            return response
    
    async def _adjust_response_for_intent(
        self,
        response: str,
        intent_analysis: Dict[str, Any],
        character_config: Dict[str, Any]
    ) -> str:
        """根据意图调整回复"""
        try:
            intent = intent_analysis.get("primary_intent", "unknown")
            sentiment = intent_analysis.get("sentiment", "neutral")
            
            # 根据意图添加适当的开头或结尾
            if intent == "greeting":
                if not any(greeting in response.lower() for greeting in ["你好", "hello", "hi"]):
                    response = "你好！" + response
            
            elif intent == "farewell":
                if not any(farewell in response.lower() for farewell in ["再见", "bye", "goodbye"]):
                    response = response + " 再见！"
            
            elif intent == "question" and not response.strip().endswith(('。', '.', '！', '!')):
                response += "。"
            
            # 根据情感调整语调
            if sentiment == "negative":
                # 对于负面情感，使用更温和、同理心的语调
                if not any(empathy in response.lower() for empathy in ["理解", "抱歉", "遗憾"]):
                    response = "我理解您的感受。" + response
            
            elif sentiment == "positive":
                # 对于正面情感，可以更热情一些
                character_traits = character_config.get("personality", {}).get("traits", [])
                if "热情" in character_traits or "活泼" in character_traits:
                    if not response.endswith(('!', '！')):
                        response = response.rstrip('。.') + '！'
            
            return response
            
        except Exception as e:
            logger.error(f"根据意图调整回复失败: {e}")
            return response
    
    async def _update_memories(
        self,
        user_id: UUID,
        session_id: UUID,
        user_input: str,
        ai_response: str,
        context: ConversationContext
    ) -> None:
        """更新记忆"""
        try:
            # 创建当前对话的消息列表
            current_messages = [
                ChatMessage(
                    id=UUID('00000000-0000-0000-0000-000000000000'),
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=user_input
                ),
                ChatMessage(
                    id=UUID('00000000-0000-0000-0000-000000000001'),
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=ai_response
                )
            ]
            
            # 从对话中提取记忆
            new_memories = await self.memory_manager.extract_memories_from_conversation(
                user_id=user_id,
                session_id=session_id,
                messages=context.messages + current_messages
            )
            
            # 存储新记忆
            for memory in new_memories:
                await self.memory_manager.store_memory(memory)
            
            logger.info(f"更新了 {len(new_memories)} 条记忆")
            
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
    
    async def _generate_error_response(
        self,
        error: str,
        character_config: Dict[str, Any]
    ) -> str:
        """生成错误处理响应"""
        try:
            character_name = character_config.get("name", "AI助手")
            
            # 根据角色特征生成不同风格的错误响应
            personality_traits = character_config.get("personality", {}).get("traits", [])
            
            if "可爱" in personality_traits:
                return f"哎呀，{character_name}遇到了一点小问题呢 😅 请稍后再试试吧！"
            elif "专业" in personality_traits:
                return f"抱歉，系统当前遇到技术问题，请稍后重试。如问题持续存在，请联系技术支持。"
            else:
                return f"抱歉，{character_name}暂时无法回复您的消息，请稍后再试。"
                
        except Exception:
            return "抱歉，系统遇到了问题，请稍后重试。"
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {}
            }
            
            # 检查Ollama服务
            if not self.ollama_service:
                await self._init_ai_clients()
            
            if self.ollama_service:
                ollama_health = await self.ollama_service.health_check()
                health_status["components"]["ollama"] = ollama_health
                if ollama_health["status"] != "healthy":
                    health_status["status"] = "degraded"
            else:
                health_status["components"]["ollama"] = "not_configured"
            
            # 检查OpenAI客户端
            if self.openai_client:
                try:
                    # 简单的API测试
                    test_response = await self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    health_status["components"]["openai"] = "healthy"
                except Exception as e:
                    health_status["components"]["openai"] = f"error: {str(e)}"
                    health_status["status"] = "degraded"
            else:
                health_status["components"]["openai"] = "not_configured"
            
            # 检查各个管理器
            health_status["components"]["context_manager"] = "healthy"
            health_status["components"]["memory_manager"] = "healthy"
            health_status["components"]["character_system"] = "healthy"
            health_status["components"]["stream_handler"] = "healthy"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# 全局AI流水线实例
ai_pipeline = AIPipeline()
