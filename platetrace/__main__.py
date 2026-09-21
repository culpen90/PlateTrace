import argparse

import uvicorn
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="PlateTrace autonomous public-vehicle research")
    parser.add_argument("--port", type=int, default=8741)
    args = parser.parse_args()
    load_dotenv()
    print(f"PlateTrace: http://127.0.0.1:{args.port}")
    uvicorn.run("platetrace.server:app", host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
