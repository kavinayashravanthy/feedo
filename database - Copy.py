from pymongo import MongoClient

MONGO_URI = "mongodb+srv://kavinayashravanthy_db_user:kavinaya2007@feedo.xetbc1f.mongodb.net/sample_mflix?retryWrites=true&w=majority&appName=feedo"

client = MongoClient(MONGO_URI)

# Choose the database
db = client["sample_mflix"]

# Example collection (you can change this later)
feedback_collection = db["feedbacks"]
