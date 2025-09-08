from pymongo import MongoClient
import os

# Your MongoDB connection string
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://kavinayashravanthy_db_user:kavinaya2007@feedo.xetbc1f.mongodb.net/sample_mflix?retryWrites=true&w=majority"
)

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["sample_mflix"]
feedback_collection = db["feedback"]

# Clear all feedback documents
result = feedback_collection.delete_many({})

print(f"✅ Deleted {result.deleted_count} feedback records.")
