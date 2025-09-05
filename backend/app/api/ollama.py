"""
Ollama API路由
提供Ollama模型管理和测试的API接口
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..services.ollama_service import get_ollama_service, OllamaService
from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ollama", tags=["Ollama"])


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    size: int
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    details: Dict[str, Any] = {}


class PullModelRequest(BaseModel):
    """拉取模型请求"""
    model_name: str


class BenchmarkRequest(BaseModel):
    """性能测试请求"""
    model_name: str
    test_prompt: Optional[str] = "你好，请介绍一下你自己。"


class ChatRequest(BaseModel):
    """聊天请求"""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.9
    stream: bool = False


@router.get("/health")
async def health_check(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """Ollama健康检查"""
    try:
        return await ollama_service.health_check()
    except Exception as e:
        logger.error(f"Ollama健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@router.get("/models")
async def list_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, List[str]]:
    """获取可用模型列表"""
    try:
        models = ollama_service.get_available_models()
        return {
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.get("/models/{model_name}")
async def get_model_info(
    model_name: str,
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> ModelInfo:
    """获取指定模型的详细信息"""
    try:
        # 检查模型是否存在
        if not ollama_service.is_model_available(model_name):
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
        
        # 获取缓存的模型信息
        model_info = ollama_service.get_model_info(model_name)
        
        if model_info:
            return ModelInfo(
                name=model_name,
                size=model_info.get("size", 0),
                modified_at=model_info.get("modified_at"),
                digest=model_info.get("digest"),
                details=model_info.get("details", {})
            )
        
        # 如果缓存中没有，尝试获取详细信息
        detailed_info = await ollama_service.show_model_info(model_name)
        if detailed_info:
            return ModelInfo(
                name=model_name,
                size=detailed_info.get("size", 0),
                modified_at=detailed_info.get("modified_at"),
                digest=detailed_info.get("digest"),
                details=detailed_info.get("details", {})
            )
        
        # 如果都没有，返回基本信息
        return ModelInfo(name=model_name, size=0)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@router.post("/models/pull")
async def pull_model(
    request: PullModelRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """拉取模型"""
    try:
        logger.info(f"开始拉取模型: {request.model_name}")
        
        success = await ollama_service.pull_model(request.model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"模型 {request.model_name} 拉取成功",
                "model_name": request.model_name
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"模型 {request.model_name} 拉取失败"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"拉取模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"拉取模型失败: {str(e)}")


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """删除模型"""
    try:
        # 检查模型是否存在
        if not ollama_service.is_model_available(model_name):
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
        
        success = await ollama_service.delete_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"模型 {model_name} 删除成功",
                "model_name": model_name
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"模型 {model_name} 删除失败"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")


@router.post("/models/refresh")
async def refresh_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """刷新模型列表"""
    try:
        await ollama_service.refresh_available_models()
        models = ollama_service.get_available_models()
        
        return {
            "status": "success",
            "message": "模型列表刷新成功",
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        logger.error(f"刷新模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"刷新模型列表失败: {str(e)}")


@router.get("/models/recommended")
async def get_recommended_models(
    task_type: str = "chat",
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """获取推荐模型"""
    try:
        all_recommended = ollama_service.get_recommended_models()
        
        if task_type in all_recommended:
            recommended = all_recommended[task_type]
            # 检查哪些推荐模型已安装
            available_recommended = [
                model for model in recommended 
                if ollama_service.is_model_available(model)
            ]
            
            # 获取建议的模型
            suggested_model = await ollama_service.suggest_model_for_task(task_type)
            
            return {
                "task_type": task_type,
                "recommended_models": recommended,
                "available_recommended": available_recommended,
                "suggested_model": suggested_model,
                "all_available": ollama_service.get_available_models()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的任务类型: {task_type}。支持的类型: {list(all_recommended.keys())}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取推荐模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取推荐模型失败: {str(e)}")


@router.post("/models/{model_name}/benchmark")
async def benchmark_model(
    model_name: str,
    request: BenchmarkRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """测试模型性能"""
    try:
        # 检查模型是否存在
        if not ollama_service.is_model_available(model_name):
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不存在")
        
        benchmark_result = await ollama_service.benchmark_model(
            model_name=model_name,
            test_prompt=request.test_prompt
        )
        
        return benchmark_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"性能测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"性能测试失败: {str(e)}")


@router.post("/chat")
async def chat_with_ollama(
    request: ChatRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """直接与Ollama模型对话（用于测试）"""
    try:
        # 检查模型是否存在
        if not ollama_service.is_model_available(request.model):
            raise HTTPException(status_code=404, detail=f"模型 {request.model} 不存在")
        
        if request.stream:
            # 流式响应需要使用WebSocket或SSE，这里不支持
            raise HTTPException(
                status_code=400,
                detail="流式响应请使用WebSocket接口"
            )
        
        response = await ollama_service.generate_response(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )
        
        return {
            "model": request.model,
            "response": response,
            "message_count": len(request.messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")


@router.get("/status")
async def get_ollama_status(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """获取Ollama服务状态"""
    try:
        health = await ollama_service.health_check()
        models = ollama_service.get_available_models()
        
        return {
            "health": health,
            "models_count": len(models),
            "available_models": models[:5],  # 只返回前5个模型
            "base_url": ollama_service.base_url,
            "service_status": "running" if health["status"] == "healthy" else "error"
        }
        
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")
