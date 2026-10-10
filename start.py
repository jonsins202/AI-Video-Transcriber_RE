#!/usr/bin/env python3
"""
AI视频转录器启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

# Windows GBK 控制台下，含 emoji 的 print 会触发 UnicodeEncodeError 直接崩溃，
# 这里把 stdout/stderr 强制成 UTF-8（无法表示的字符替换为 ?，绝不抛异常）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 加载项目根 .env（已有环境变量优先，不覆盖），让 PORT / OPENAI_* 等对启动脚本同样生效
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

def check_dependencies():
    """检查依赖是否安装"""
    import sys
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "yt-dlp": "yt_dlp",
        "funasr": "funasr",
        "qwen-asr": "qwen_asr",
        "openai": "openai"
    }
    
    missing_packages = []
    for display_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(display_name)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("source venv/bin/activate && pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖已安装")
    return True

def check_ffmpeg():
    """检查FFmpeg是否安装"""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        print("✅ FFmpeg已安装")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 未找到FFmpeg")
        print("请安装FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: 从官网下载 https://ffmpeg.org/download.html")
        return False

def setup_environment():
    """设置环境变量"""
    # 设置OpenAI配置
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  警告: 未设置OPENAI_API_KEY环境变量")
        print("请设置环境变量: export OPENAI_API_KEY=your_api_key_here")
        return False
    
    print("✅ 已设置OpenAI API Key")
    
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://oneapi.basevec.com/v1"
        print("✅ 已设置OpenAI Base URL")
    
    # 设置 Qwen3-ASR 模型路径（备选后端）
    if not os.getenv("QWEN3_ASR_MODEL_PATH"):
        default_model = str(Path(__file__).parent / "models" / "Qwen3-ASR-0.6B")
        os.environ["QWEN3_ASR_MODEL_PATH"] = default_model

    # ASR 后端默认配置（FunASR 为默认；ASR_BACKEND=qwen 切本地 Qwen3-ASR，dashscope 切云端）
    os.environ.setdefault("ASR_BACKEND", "funasr")
    os.environ.setdefault("FUNASR_MODEL", "paraformer")
    os.environ.setdefault("FUNASR_DEVICE", "cuda")
    os.environ.setdefault("FUNASR_VAD_MODEL", "fsmn-vad")
    os.environ.setdefault("FUNASR_BATCH_SIZE_S", "300")
    # 摘要分块并发数
    os.environ.setdefault("SUMMARY_CONCURRENCY", "6")
    # 无损优化并发数（DeepSeek 账号级并发上限 500，32 并发安全；受 asyncio 线程池 min(32,cpu+4) 限制）
    os.environ.setdefault("LOSSLESS_CONCURRENCY", "32")
    _be = os.getenv("ASR_BACKEND", "funasr")
    _detail = f" ({os.getenv('FUNASR_MODEL')})" if _be == "funasr" else ""
    print(f"🎙  ASR 后端: {_be}{_detail}")
    print(f"⚙  并发: 无损优化={os.getenv('LOSSLESS_CONCURRENCY')}  摘要={os.getenv('SUMMARY_CONCURRENCY')}")

    print("🔑 OpenAI API已配置，摘要功能可用")
    return True

def main():
    """主函数"""
    # 检查是否使用生产模式（禁用热重载）
    production_mode = "--prod" in sys.argv or os.getenv("PRODUCTION_MODE") == "true"
    
    print("🚀 AI视频转录器启动检查")
    if production_mode:
        print("🔒 生产模式 - 热重载已禁用")
    else:
        print("🔧 开发模式 - 热重载已启用")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查FFmpeg
    if not check_ffmpeg():
        print("⚠️  FFmpeg未安装，可能影响某些视频格式的处理")
    
    # 设置环境
    setup_environment()
    
    print("\n🎉 启动检查完成!")
    print("=" * 50)
    
    # 启动服务器
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 9000))
    if "--port" in sys.argv:  # 临时覆盖：python start.py --prod --port 9001
        port = int(sys.argv[sys.argv.index("--port") + 1])
    
    print(f"\n🌐 启动服务器...")
    print(f"   地址: http://localhost:{port}")
    print(f"   按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        # 切换到backend目录并启动服务
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", host,
            "--port", str(port)
        ]
        
        # 只在开发模式下启用热重载
        if not production_mode:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
