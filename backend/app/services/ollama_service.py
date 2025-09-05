"""
Ollama AI服务
提供与Ollama本地AI模型的交互功能
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
import aiohttp
import ollama
from ollama import AsyncClient

from ..config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class OllamaService:
    """Ollama AI服务类"""
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.client = AsyncClient(host=self.base_url)
        self.available_models = []
        self.model_info_cache = {}
        
    async def initialize(self):
        """初始化服务，检查连接和获取可用模型"""
        try:
            # 检查Ollama服务是否运行
            await self.health_check()
            
            # 获取可用模型列表
            await self.refresh_available_models()
            
            logger.info(f"Ollama服务初始化成功，可用模型: {len(self.available_models)}个")
            return True
            
        except Exception as e:
            logger.error(f"Ollama服务初始化失败: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "base_url": self.base_url,
                            "response_time": response.headers.get("response-time", "unknown")
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}",
                            "base_url": self.base_url
                        }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "base_url": self.base_url
            }
    
    async def refresh_available_models(self):
        """刷新可用模型列表"""
        try:
            models_response = await self.client.list()
            self.available_models = [model['name'] for model in models_response.get('models', [])]
            
            # 更新模型信息缓存
            for model in models_response.get('models', []):
                self.model_info_cache[model['name']] = {
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at'),
                    'digest': model.get('digest'),
                    'details': model.get('details', {})
                }
                
            logger.info(f"已刷新模型列表，共{len(self.available_models)}个模型")
            
        except Exception as e:
            logger.error(f"刷新模型列表失败: {e}")
            self.available_models = []
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return self.available_models.copy()
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        return self.model_info_cache.get(model_name)
    
    async def pull_model(self, model_name: str) -> bool:
        """拉取模型"""
        try:
            logger.info(f"开始拉取模型: {model_name}")
            
            # 使用流式拉取以显示进度
            async for progress in await self.client.pull(model_name, stream=True):
                if 'status' in progress:
                    logger.info(f"拉取进度: {progress['status']}")
                    
            logger.info(f"模型 {model_name} 拉取完成")
            
            # 刷新模型列表
            await self.refresh_available_models()
            
            return True
            
        except Exception as e:
            logger.error(f"拉取模型 {model_name} 失败: {e}")
            return False
    
    async def generate_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> str:
        """生成AI响应（非流式）"""
        try:
            if model not in self.available_models:
                raise ValueError(f"模型 {model} 不可用，可用模型: {self.available_models}")
            
            # 构建请求参数
            request_params = {
                'model': model,
                'messages': messages,
                'stream': stream,
                'options': {
                    'temperature': temperature,
                    'top_p': top_p,
                }
            }
            
            # 添加max_tokens参数（如果支持）
            if max_tokens:
                request_params['options']['num_predict'] = max_tokens
            
            # 添加其他参数
            for key, value in kwargs.items():
                if key in ['repeat_penalty', 'seed', 'stop']:
                    request_params['options'][key] = value
            
            logger.debug(f"发送请求到Ollama: {model}")
            
            response = await self.client.chat(**request_params)
            
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                logger.error(f"Ollama响应格式异常: {response}")
                return "抱歉，生成响应时出现问题。"
                
        except Exception as e:
            logger.error(f"Ollama生成响应失败: {e}")
            return f"抱歉，处理您的请求时出现问题: {str(e)}"
    
    async def generate_streaming_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """生成流式AI响应"""
        try:
            if model not in self.available_models:
                raise ValueError(f"模型 {model} 不可用，可用模型: {self.available_models}")
            
            # 构建请求参数
            request_params = {
                'model': model,
                'messages': messages,
                'stream': True,
                'options': {
                    'temperature': temperature,
                    'top_p': top_p,
                }
            }
            
            # 添加max_tokens参数（如果支持）
            if max_tokens:
                request_params['options']['num_predict'] = max_tokens
            
            # 添加其他参数
            for key, value in kwargs.items():
                if key in ['repeat_penalty', 'seed', 'stop']:
                    request_params['options'][key] = value
            
            logger.debug(f"开始流式生成: {model}")
            
            async for chunk in await self.client.chat(**request_params):
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:  # 只返回非空内容
                        yield content
                        
        except Exception as e:
            logger.error(f"Ollama流式生成失败: {e}")
            yield f"抱歉，处理您的请求时出现问题: {str(e)}"
    
    async def generate_embeddings(
        self,
        model: str,
        text: str
    ) -> Optional[List[float]]:
        """生成文本嵌入向量"""
        try:
            if model not in self.available_models:
                # 尝试检查是否为嵌入模型
                embedding_models = [m for m in self.available_models if 'embed' in m.lower()]
                if not embedding_models:
                    logger.warning(f"未找到嵌入模型，可用模型: {self.available_models}")
                    return None
                model = embedding_models[0]  # 使用第一个找到的嵌入模型
            
            response = await self.client.embeddings(
                model=model,
                prompt=text
            )
            
            if 'embedding' in response:
                return response['embedding']
            else:
                logger.error(f"嵌入响应格式异常: {response}")
                return None
                
        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            return None
    
    async def show_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取详细模型信息"""
        try:
            response = await self.client.show(model_name)
            return response
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return None
    
    async def delete_model(self, model_name: str) -> bool:
        """删除模型"""
        try:
            await self.client.delete(model_name)
            logger.info(f"已删除模型: {model_name}")
            
            # 刷新模型列表
            await self.refresh_available_models()
            
            return True
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False
    
    def is_model_available(self, model_name: str) -> bool:
        """检查模型是否可用"""
        return model_name in self.available_models
    
    def get_recommended_models(self) -> Dict[str, List[str]]:
        """获取推荐模型列表"""
        return {
            "chat": [
                "llama2:7b", "llama2:13b", "llama2:70b",
                "mistral:7b", "mixtral:8x7b",
                "codellama:7b", "codellama:13b",
                "qwen:7b", "qwen:14b"
            ],
            "embedding": [
                "nomic-embed-text",
                "all-minilm:l6-v2",
                "all-minilm:l12-v2"
            ],
            "code": [
                "codellama:7b", "codellama:13b",
                "starcoder:7b", "starcoder:15b"
            ]
        }
    
    async def suggest_model_for_task(self, task_type: str = "chat") -> Optional[str]:
        """根据任务类型推荐合适的模型"""
        recommended = self.get_recommended_models().get(task_type, [])
        
        # 找到第一个可用的推荐模型
        for model in recommended:
            if self.is_model_available(model):
                return model
        
        # 如果没有推荐模型可用，返回第一个可用模型
        if self.available_models:
            return self.available_models[0]
        
        return None
    
    async def benchmark_model(self, model_name: str, test_prompt: str = "你好，请介绍一下你自己。") -> Dict[str, Any]:
        """测试模型性能"""
        if not self.is_model_available(model_name):
            return {"error": f"模型 {model_name} 不可用"}
        
        try:
            start_time = time.time()
            
            # 测试简单对话
            messages = [{"role": "user", "content": test_prompt}]
            response = await self.generate_response(
                model=model_name,
                messages=messages,
                temperature=0.7
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 计算基本统计
            response_length = len(response)
            tokens_per_second = response_length / response_time if response_time > 0 else 0
            
            return {
                "model": model_name,
                "response_time": round(response_time, 2),
                "response_length": response_length,
                "tokens_per_second": round(tokens_per_second, 2),
                "test_prompt": test_prompt,
                "response_preview": response[:100] + "..." if len(response) > 100 else response,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "model": model_name,
                "error": str(e),
                "status": "failed"
            }


# 全局Ollama服务实例
ollama_service = OllamaService()


async def get_ollama_service() -> OllamaService:
    """获取Ollama服务实例"""
    if not ollama_service.available_models:
        await ollama_service.initialize()
    return ollama_service
