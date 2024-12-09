# run_server.py

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="whisper api")
    parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default=8080, help="default: 8080")
    parser.add_argument("-w", "--workers", type=int, default=4, help="default: 4")
    args = parser.parse_args()

    uvicorn.run(
        "whisper_api:app",
        host=args.bind_addr,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
