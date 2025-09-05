"""
FastAPI主应用程序

AI聊天机器人系统的Web服务入口
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import get_settings
from .core.ai_pipeline import ai_pipeline
from .core.stream_handler import stream_handler
from .models.chat_models import MessageCreate, MessageResponse, SessionCreate, SessionResponse
from .models.user_models import UserCreate, UserResponse, UserLogin, TokenResponse
from .utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("启动AI聊天机器人系统")
    
    # 健康检查
    health_status = await ai_pipeline.health_check()
    if health_status["status"] != "healthy":
        logger.warning(f"系统健康检查警告: {health_status}")
    
    # 启动后台任务
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # 关闭时执行
    logger.info("关闭AI聊天机器人系统")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# 创建FastAPI应用
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# 全局异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "内部服务器错误",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )


# 根路由
@app.get("/")
async def root():
    """根路由"""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        health_status = await ai_pipeline.health_check()
        
        if health_status["status"] == "healthy":
            return health_status
        else:
            return JSONResponse(
                status_code=503,
                content=health_status
            )
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# 系统信息
@app.get("/info")
async def system_info():
    """获取系统信息"""
    return {
        "system": {
            "name": settings.api_title,
            "version": settings.api_version,
            "environment": settings.environment,
            "debug": settings.debug
        },
        "features": {
            "streaming": settings.ai.streaming.enabled,
            "live2d": settings.live2d.enabled,
            "voice": True,
            "memory": True,
            "characters": len(ai_pipeline.character_system.list_characters())
        },
        "stats": {
            "active_streams": stream_handler.get_active_stream_count(),
            "stream_stats": stream_handler.get_stream_stats()
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# 角色管理API
@app.get("/api/characters")
async def list_characters():
    """获取可用角色列表"""
    try:
        characters = ai_pipeline.character_system.list_characters()
        return {"characters": characters}
    except Exception as e:
        logger.error(f"获取角色列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取角色列表失败")


@app.get("/api/characters/{character_id}")
async def get_character(character_id: str):
    """获取角色详情"""
    try:
        character = ai_pipeline.character_system.get_character(character_id)
        if not character:
            raise HTTPException(status_code=404, detail="角色不存在")
        
        # 移除敏感信息
        safe_character = {
            "id": character_id,
            "name": character["name"],
            "description": character["description"],
            "personality": character["personality"],
            "knowledge_domains": character["knowledge_domains"]
        }
        
        return {"character": safe_character}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取角色详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取角色详情失败")


# 聊天API
@app.post("/api/chat/message")
async def send_message(
    message: MessageCreate,
    session_id: str,
    user_id: str = "anonymous",
    character_id: str = "default"
):
    """发送聊天消息（非流式）"""
    try:
        from uuid import UUID
        
        # 验证UUID格式
        try:
            session_uuid = UUID(session_id)
            user_uuid = UUID(user_id) if user_id != "anonymous" else UUID('00000000-0000-0000-0000-000000000000')
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的ID格式")
        
        # 处理消息
        response_text, metadata = await ai_pipeline.process_message(
            user_id=user_uuid,
            session_id=session_uuid,
            message=message,
            character_id=character_id
        )
        
        # 构建响应
        response = MessageResponse(
            id=UUID('00000000-0000-0000-0000-000000000001'),
            session_id=session_uuid,
            role="assistant",
            content=response_text,
            message_type="text",
            status="sent",
            metadata=metadata,
            created_at=datetime.utcnow()
        )
        
        return {"message": response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"发送消息失败: {e}")
        raise HTTPException(status_code=500, detail="发送消息失败")


# WebSocket聊天端点
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str, user_id: str = "anonymous", character_id: str = "default"):
    """WebSocket聊天端点"""
    await websocket.accept()
    stream_id = None
    
    try:
        from uuid import UUID
        
        # 验证UUID格式
        try:
            session_uuid = UUID(session_id)
            user_uuid = UUID(user_id) if user_id != "anonymous" else UUID('00000000-0000-0000-0000-000000000000')
        except ValueError:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "无效的ID格式"}
            })
            return
        
        # 创建流式连接
        stream_id = await stream_handler.create_stream(
            session_id=session_uuid,
            websocket=websocket,
            user_id=user_uuid
        )
        
        logger.info(f"WebSocket连接已建立: {stream_id}")
        
        # 发送连接确认
        await websocket.send_json({
            "type": "connection_established",
            "data": {
                "stream_id": stream_id,
                "session_id": session_id,
                "character_id": character_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        # 消息处理循环
        while True:
            try:
                # 接收消息
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "message":
                    # 处理聊天消息
                    content = data.get("content", "").strip()
                    if not content:
                        continue
                    
                    # 发送打字指示器
                    await stream_handler.handle_typing_indicator(stream_id, True)
                    
                    # 创建消息对象
                    message = MessageCreate(content=content)
                    
                    # 处理消息（流式）
                    response_text, metadata = await ai_pipeline.process_message(
                        user_id=user_uuid,
                        session_id=session_uuid,
                        message=message,
                        character_id=character_id,
                        stream_id=stream_id
                    )
                    
                    # 停止打字指示器
                    await stream_handler.handle_typing_indicator(stream_id, False)
                    
                    # 发送完成通知
                    await websocket.send_json({
                        "type": "message_complete",
                        "data": {
                            "metadata": metadata,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })
                
                elif message_type == "interrupt":
                    # 中断当前响应
                    await stream_handler.interrupt_stream(stream_id)
                
                elif message_type == "ping":
                    # 心跳响应
                    await websocket.send_json({
                        "type": "pong",
                        "data": {"timestamp": datetime.utcnow().isoformat()}
                    })
                
                elif message_type == "change_character":
                    # 更换角色
                    new_character_id = data.get("character_id", "default")
                    character = ai_pipeline.character_system.get_character(new_character_id)
                    if character:
                        character_id = new_character_id
                        await websocket.send_json({
                            "type": "character_changed",
                            "data": {
                                "character_id": character_id,
                                "character_name": character["name"]
                            }
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"message": "角色不存在"}
                        })
                
                else:
                    logger.warning(f"未知消息类型: {message_type}")
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"处理WebSocket消息失败: {e}")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"处理消息失败: {str(e)}"}
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {stream_id}")
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    finally:
        # 清理资源
        if stream_id:
            await stream_handler.close_stream(stream_id)


# 会话管理API
@app.post("/api/sessions")
async def create_session(session: SessionCreate, user_id: str = "anonymous"):
    """创建新会话"""
    try:
        from uuid import UUID, uuid4
        
        user_uuid = UUID(user_id) if user_id != "anonymous" else UUID('00000000-0000-0000-0000-000000000000')
        session_uuid = uuid4()
        
        # 这里应该保存到数据库
        response = SessionResponse(
            id=session_uuid,
            user_id=user_uuid,
            title=session.title,
            status="active",
            character_config=session.character_config,
            ai_model=session.ai_model,
            temperature=session.temperature,
            created_at=datetime.utcnow()
        )
        
        return {"session": response}
        
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        raise HTTPException(status_code=500, detail="创建会话失败")


# 统计信息API
@app.get("/api/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        return {
            "system": {
                "uptime": "运行中",
                "version": settings.api_version,
                "environment": settings.environment
            },
            "streaming": stream_handler.get_stream_stats(),
            "ai": {
                "model": settings.ai.model.name,
                "provider": settings.ai.model.provider
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")


# 后台清理任务
async def periodic_cleanup():
    """定期清理任务"""
    while True:
        try:
            # 清理不活跃的流连接
            await stream_handler.cleanup_inactive_streams()
            
            # 清理过期记忆
            await ai_pipeline.memory_manager.cleanup_expired_memories()
            
            logger.debug("定期清理任务完成")
            
            # 每5分钟执行一次
            await asyncio.sleep(300)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"定期清理任务失败: {e}")
            await asyncio.sleep(60)  # 出错时等待1分钟再重试


if __name__ == "__main__":
    # 开发环境启动
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development(),
        log_level=settings.monitoring.log_level.lower(),
        access_log=settings.is_development()
    )
