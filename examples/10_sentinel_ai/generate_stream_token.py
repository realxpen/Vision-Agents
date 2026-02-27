import argparse
import os

from dotenv import load_dotenv
from getstream import Stream


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate Stream browser user token.")
    parser.add_argument("--user-id", default="user-demo-agent", help="User ID")
    parser.add_argument("--name", default="Human User", help="User display name")
    parser.add_argument(
        "--expires-in-seconds", type=int, default=3600, help="Token TTL in seconds"
    )
    args = parser.parse_args()

    api_key = os.getenv("STREAM_API_KEY")
    api_secret = os.getenv("STREAM_API_SECRET")
    if not api_key or not api_secret:
        raise SystemExit("Missing STREAM_API_KEY or STREAM_API_SECRET in .env")

    client = Stream(api_key=api_key, api_secret=api_secret)
    token = client.create_token(args.user_id, expiration=args.expires_in_seconds)

    print(f"API_KEY={api_key}")
    print(f"USER_ID={args.user_id}")
    print(f"NAME={args.name}")
    print(f"TOKEN={token}")


if __name__ == "__main__":
    main()
