from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

# MongoDB connection
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/heart_prediction')
client = MongoClient(MONGO_URI)
# Extract database name from URI or use default
db_name = MONGO_URI.split('/')[-1] if '/' in MONGO_URI else 'heart_prediction'
db = client[db_name] if db_name else client.get_database('heart_prediction')

class User:
    collection = db.users
    
    @staticmethod
    def create(name, email, password):
        user = {
            "name": name,
            "email": email,
            "password": generate_password_hash(password),
            "createdAt": datetime.utcnow(),
            "theme": "light"
        }
        result = User.collection.insert_one(user)
        user["_id"] = result.inserted_id
        return user
    
    @staticmethod
    def find_by_email(email):
        return User.collection.find_one({"email": email})
    
    @staticmethod
    def find_by_id(user_id):
        from bson import ObjectId
        return User.collection.find_one({"_id": ObjectId(user_id)})
    
    @staticmethod
    def update(user_id, data):
        from bson import ObjectId
        update_data = {}
        
        if "name" in data:
            update_data["name"] = data["name"]
        if "email" in data:
            update_data["email"] = data["email"]
        if "password" in data:
            update_data["password"] = generate_password_hash(data["password"])
        if "theme" in data:
            update_data["theme"] = data["theme"]
        
        if not update_data:
            return None
        
        result = User.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return User.find_by_id(user_id)
        return None

class PredictionRecord:
    collection = db.records
    
    @staticmethod
    def create(user_id, inputs, prediction, probability):
        from bson import ObjectId
        record = {
            "userId": ObjectId(user_id),
            "inputs": inputs,
            "prediction": prediction,
            "probability": probability,
            "timestamp": datetime.utcnow()
        }
        result = PredictionRecord.collection.insert_one(record)
        record["_id"] = result.inserted_id
        return record
    
    @staticmethod
    def find_by_user(user_id):
        from bson import ObjectId
        return list(PredictionRecord.collection.find(
            {"userId": ObjectId(user_id)}
        ).sort("timestamp", -1))
    
    @staticmethod
    def count_by_user(user_id):
        from bson import ObjectId
        return PredictionRecord.collection.count_documents(
            {"userId": ObjectId(user_id)}
        )
    
    @staticmethod
    def delete(record_id, user_id):
        from bson import ObjectId
        result = PredictionRecord.collection.delete_one({
            "_id": ObjectId(record_id),
            "userId": ObjectId(user_id)
        })
        return result.deleted_count > 0

