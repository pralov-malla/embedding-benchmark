import os
import boto3
from dotenv import load_dotenv

load_dotenv()

region = os.getenv("AWS_REGION", "ap-south-1")
bedrock = boto3.client(
    service_name="bedrock",
    region_name=region,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
)

print(f"Querying available Inference Profiles in region '{region}'...\n")

try:
    response = bedrock.list_inference_profiles(typeEquals="SYSTEM_DEFINED")
    profiles = response.get("inferenceProfileSummaries", [])
    
    embed_profiles = [
        p for p in profiles 
        if "embed" in p["inferenceProfileId"].lower() or "cohere" in p["inferenceProfileId"].lower()
    ]
    
    if embed_profiles:
        print(f"Found {len(embed_profiles)} relevant profile(s):")
        for p in embed_profiles:
            print(f" - ID:  {p['inferenceProfileId']}")
            print(f"   ARN: {p['inferenceProfileArn']}\n")
    else:
        print("No specific embedding inference profiles found. All profiles:")
        for p in profiles:
            print(f" - ID: {p['inferenceProfileId']}")

except Exception as e:
    print(f"Error querying inference profiles: {e}")

