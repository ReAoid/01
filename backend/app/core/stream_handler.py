"""
流式处理系统

实现WebSocket流式响应、分块处理和实时通信功能
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator, Callable, Any
from uuid import UUID, uuid4

from ..config import get_settings
from ..models.chat_models import StreamChunk, MessageRole
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class StreamHandler:
    """流式处理器"""
    
    def __init__(self):
        self.active_streams = {}  # 活跃的流式连接
        self.chunk_buffer = {}    # 分块缓冲区
        self.interrupt_signals = {}  # 中断信号
    
    async def create_stream(
        self,
        session_id: UUID,
        websocket,
        user_id: Optional[UUID] = None
    ) -> str:
        """创建新的流式连接"""
        try:
            stream_id = str(uuid4())
            
            self.active_streams[stream_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "websocket": websocket,
                "created_at": datetime.utcnow(),
                "chunk_count": 0,
                "total_tokens": 0,
                "is_active": True
            }
            
            self.chunk_buffer[stream_id] = []
            self.interrupt_signals[stream_id] = False
            
            logger.info(f"创建流式连接: {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"创建流式连接失败: {e}")
            raise
    
    async def close_stream(self, stream_id: str):
        """关闭流式连接"""
        try:
            if stream_id in self.active_streams:
                self.active_streams[stream_id]["is_active"] = False
                
                # 清理资源
                self.chunk_buffer.pop(stream_id, None)
                self.interrupt_signals.pop(stream_id, None)
                
                # 从活跃连接中移除
                del self.active_streams[stream_id]
                
                logger.info(f"关闭流式连接: {stream_id}")
                
        except Exception as e:
            logger.error(f"关闭流式连接失败: {e}")
    
    async def send_stream_chunk(
        self,
        stream_id: str,
        content: str,
        is_final: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """发送流式分块"""
        try:
            if stream_id not in self.active_streams:
                logger.warning(f"流式连接不存在: {stream_id}")
                return False
            
            stream_info = self.active_streams[stream_id]
            
            if not stream_info["is_active"]:
                logger.warning(f"流式连接已关闭: {stream_id}")
                return False
            
            # 检查是否被中断
            if self.interrupt_signals.get(stream_id, False):
                logger.info(f"流式传输被中断: {stream_id}")
                return False
            
            # 创建分块对象
            chunk = StreamChunk(
                id=uuid4(),
                session_id=stream_info["session_id"],
                chunk_id=stream_info["chunk_count"],
                content=content,
                is_final=is_final,
                metadata=metadata
            )
            
            # 发送到WebSocket
            websocket = stream_info["websocket"]
            chunk_data = {
                "type": "stream_chunk",
                "data": chunk.dict()
            }
            
            await websocket.send_text(json.dumps(chunk_data, ensure_ascii=False))
            
            # 更新统计信息
            stream_info["chunk_count"] += 1
            stream_info["total_tokens"] += len(content.split())
            
            # 缓存分块（用于重发或调试）
            if stream_id in self.chunk_buffer:
                self.chunk_buffer[stream_id].append(chunk)
                
                # 限制缓存大小
                if len(self.chunk_buffer[stream_id]) > 100:
                    self.chunk_buffer[stream_id] = self.chunk_buffer[stream_id][-50:]
            
            logger.debug(f"发送流式分块: {stream_id}, 分块#{chunk.chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"发送流式分块失败: {e}")
            return False
    
    async def stream_ai_response(
        self,
        stream_id: str,
        response_generator: AsyncGenerator[str, None],
        character_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """流式传输AI响应"""
        try:
            if stream_id not in self.active_streams:
                return False
            
            # 获取流式配置
            chunk_size = settings.ai.streaming.chunk_size
            delay_ms = settings.ai.streaming.delay_ms
            
            current_chunk = ""
            chunk_count = 0
            
            async for token in response_generator:
                # 检查中断信号
                if self.interrupt_signals.get(stream_id, False):
                    logger.info(f"AI响应流被中断: {stream_id}")
                    break
                
                current_chunk += token
                
                # 根据配置决定何时发送分块
                should_send = False
                
                if len(current_chunk) >= chunk_size:
                    should_send = True
                elif token in ["。", "！", "？", ".", "!", "?"]:
                    # 句子结束时发送
                    should_send = True
                elif token in ["，", "；", ",", ";"]:
                    # 长句子中的暂停点
                    if len(current_chunk) > chunk_size // 2:
                        should_send = True
                
                if should_send and current_chunk.strip():
                    # 发送当前分块
                    success = await self.send_stream_chunk(
                        stream_id,
                        current_chunk,
                        is_final=False,
                        metadata={
                            "token_count": len(current_chunk.split()),
                            "chunk_type": "partial"
                        }
                    )
                    
                    if not success:
                        break
                    
                    current_chunk = ""
                    chunk_count += 1
                    
                    # 添加延迟以模拟打字效果
                    if delay_ms > 0:
                        await asyncio.sleep(delay_ms / 1000)
            
            # 发送最后的分块
            if current_chunk.strip():
                await self.send_stream_chunk(
                    stream_id,
                    current_chunk,
                    is_final=True,
                    metadata={
                        "token_count": len(current_chunk.split()),
                        "chunk_type": "final",
                        "total_chunks": chunk_count + 1
                    }
                )
            else:
                # 发送结束信号
                await self.send_stream_chunk(
                    stream_id,
                    "",
                    is_final=True,
                    metadata={
                        "chunk_type": "end",
                        "total_chunks": chunk_count
                    }
                )
            
            logger.info(f"AI响应流传输完成: {stream_id}, 总分块数: {chunk_count}")
            return True
            
        except Exception as e:
            logger.error(f"流式传输AI响应失败: {e}")
            return False
    
    async def interrupt_stream(self, stream_id: str) -> bool:
        """中断流式传输"""
        try:
            if stream_id not in self.active_streams:
                return False
            
            self.interrupt_signals[stream_id] = True
            
            # 发送中断通知
            await self.send_stream_chunk(
                stream_id,
                "",
                is_final=True,
                metadata={
                    "chunk_type": "interrupted",
                    "reason": "用户中断"
                }
            )
            
            logger.info(f"流式传输已中断: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"中断流式传输失败: {e}")
            return False
    
    def is_stream_active(self, stream_id: str) -> bool:
        """检查流是否活跃"""
        return (stream_id in self.active_streams and 
                self.active_streams[stream_id]["is_active"])
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流信息"""
        return self.active_streams.get(stream_id)
    
    async def handle_typing_indicator(
        self,
        stream_id: str,
        is_typing: bool
    ) -> bool:
        """处理打字指示器"""
        try:
            if stream_id not in self.active_streams:
                return False
            
            stream_info = self.active_streams[stream_id]
            websocket = stream_info["websocket"]
            
            typing_data = {
                "type": "typing_indicator",
                "data": {
                    "is_typing": is_typing,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            await websocket.send_text(json.dumps(typing_data, ensure_ascii=False))
            return True
            
        except Exception as e:
            logger.error(f"处理打字指示器失败: {e}")
            return False
    
    async def send_status_update(
        self,
        stream_id: str,
        status: str,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """发送状态更新"""
        try:
            if stream_id not in self.active_streams:
                return False
            
            stream_info = self.active_streams[stream_id]
            websocket = stream_info["websocket"]
            
            status_data = {
                "type": "status_update",
                "data": {
                    "status": status,
                    "message": message,
                    "data": data or {},
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            await websocket.send_text(json.dumps(status_data, ensure_ascii=False))
            return True
            
        except Exception as e:
            logger.error(f"发送状态更新失败: {e}")
            return False
    
    async def broadcast_to_session(
        self,
        session_id: UUID,
        message_type: str,
        data: Dict[str, Any]
    ) -> int:
        """向会话中的所有连接广播消息"""
        try:
            broadcast_count = 0
            
            for stream_id, stream_info in self.active_streams.items():
                if (stream_info["session_id"] == session_id and 
                    stream_info["is_active"]):
                    
                    try:
                        websocket = stream_info["websocket"]
                        message_data = {
                            "type": message_type,
                            "data": data
                        }
                        
                        await websocket.send_text(json.dumps(message_data, ensure_ascii=False))
                        broadcast_count += 1
                        
                    except Exception as e:
                        logger.warning(f"广播到流 {stream_id} 失败: {e}")
                        # 标记连接为不活跃
                        stream_info["is_active"] = False
            
            return broadcast_count
            
        except Exception as e:
            logger.error(f"广播消息失败: {e}")
            return 0
    
    async def cleanup_inactive_streams(self):
        """清理不活跃的流连接"""
        try:
            current_time = datetime.utcnow()
            inactive_streams = []
            
            for stream_id, stream_info in self.active_streams.items():
                # 检查连接是否超时（30分钟）
                time_diff = current_time - stream_info["created_at"]
                if time_diff.total_seconds() > 1800:  # 30分钟
                    inactive_streams.append(stream_id)
                    continue
                
                # 检查WebSocket连接状态
                try:
                    websocket = stream_info["websocket"]
                    if websocket.client_state.name != "CONNECTED":
                        inactive_streams.append(stream_id)
                except:
                    inactive_streams.append(stream_id)
            
            # 清理不活跃的连接
            for stream_id in inactive_streams:
                await self.close_stream(stream_id)
            
            if inactive_streams:
                logger.info(f"清理了 {len(inactive_streams)} 个不活跃的流连接")
            
        except Exception as e:
            logger.error(f"清理不活跃流连接失败: {e}")
    
    def get_active_stream_count(self) -> int:
        """获取活跃流连接数量"""
        return len([s for s in self.active_streams.values() if s["is_active"]])
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """获取流处理统计信息"""
        try:
            total_streams = len(self.active_streams)
            active_streams = self.get_active_stream_count()
            
            total_chunks = sum(s["chunk_count"] for s in self.active_streams.values())
            total_tokens = sum(s["total_tokens"] for s in self.active_streams.values())
            
            return {
                "total_streams": total_streams,
                "active_streams": active_streams,
                "total_chunks_sent": total_chunks,
                "total_tokens_streamed": total_tokens,
                "buffer_size": sum(len(buffer) for buffer in self.chunk_buffer.values()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取流统计信息失败: {e}")
            return {}


# 智能分块器
class SmartChunker:
    """智能分块器，根据内容语义进行分块"""
    
    def __init__(self):
        self.sentence_endings = ["。", "！", "？", ".", "!", "?"]
        self.clause_separators = ["，", "；", "：", ",", ";", ":"]
        self.natural_breaks = ["，", "；", "：", "、", ",", ";", ":", "·"]
    
    def chunk_text(
        self,
        text: str,
        max_chunk_size: int = 50,
        min_chunk_size: int = 10
    ) -> List[str]:
        """智能文本分块"""
        if not text:
            return []
        
        chunks = []
        current_chunk = ""
        
        for char in text:
            current_chunk += char
            
            # 检查是否应该分块
            should_chunk = False
            
            if len(current_chunk) >= max_chunk_size:
                should_chunk = True
            elif char in self.sentence_endings:
                should_chunk = True
            elif (char in self.clause_separators and 
                  len(current_chunk) >= min_chunk_size):
                should_chunk = True
            
            if should_chunk and current_chunk.strip():
                chunks.append(current_chunk)
                current_chunk = ""
        
        # 添加剩余内容
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_by_semantic_units(self, text: str) -> List[str]:
        """按语义单元分块"""
        # 简单的语义分块实现
        # 实际应用中可以使用更复杂的NLP技术
        
        chunks = []
        sentences = self._split_sentences(text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) > 100:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in self.sentence_endings:
                sentences.append(current_sentence)
                current_sentence = ""
        
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences


# 全局流处理器实例
stream_handler = StreamHandler()
