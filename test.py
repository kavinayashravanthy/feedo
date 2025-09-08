from pymongo import MongoClient

uri = "mongodb+srv://kavinayashravanthy_db_user:kavinaya2007@feedo.xetbc1f.mongodb.net/sample_mflix?retryWrites=true&w=majority&appName=feedo"
client = MongoClient(uri)

print(client.list_database_names())
