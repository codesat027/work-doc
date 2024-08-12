import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from pymilvus.exceptions import IndexNotExistException
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Step 1: Load the ResNet50 Model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Step 2: Connect to Milvus
connections.connect("default", host='localhost', port='19530')  # Adjust host and port as necessary

# Step 3: Define the Milvus Collection Schema
fields = [
    FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Primary key field
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048),  # ResNet50 output dimension
    FieldSchema(name="image_id", dtype=DataType.INT64)  # Optional field for image ID
]
schema = CollectionSchema(fields, description="Facial embeddings collection")
collection_name = "facial_embeddings"

# Create the collection if it doesn't exist
if collection_name not in utility.list_collections():
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# Step 4: Function to Extract Features
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()  # Flatten the features to 1D

# Step 5: Insert Features into Milvus
def insert_features(image_folder):
    embeddings = []
    ids = []
    filenames = []  # List to store actual filenames

    print("Starting feature extraction...")
    for img_id, img_file in enumerate(os.listdir(image_folder)):
        if img_file.endswith(('.jpg', '.png')):  # Adjust as necessary
            img_path = os.path.join(image_folder, img_file)
            features = extract_features(img_path)
            embeddings.append(features)
            ids.append(img_id)  # Use the index as the image ID
            filenames.append(img_file)  # Store the filename
            print(f"Extracted features from {img_file} ({img_id + 1}/{len(os.listdir(image_folder))})")

    # Insert into Milvus
    print("Inserting features into Milvus...")
    collection.insert([embeddings, ids])
    print("Insertion complete.")

    # Create an index for the embedding field if it doesn't exist
    try:
        # Attempt to retrieve index information
        index_info = collection.index()
        print("Index already exists.")
    except IndexNotExistException:
        # If the index does not exist, create it
        index_params = {
            "index_type": "IVF_FLAT",  # Choose the appropriate index type
            "metric_type": "L2",        # Choose the appropriate metric type
            "params": {"nlist": 100}    # Adjust nlist as needed
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Index created.")

    # Load the collection after creating the index
    collection.load()
    print("Collection loaded.")

    return filenames  # Return the list of filenames

# Step 6: Find Similar Images
def find_similar_images(user_image_path, top_k=3):
    # Load the collection to ensure it is ready for searching
    collection.load()  # Load the collection

    user_features = extract_features(user_image_path)
    search_params = {
        "metric_type": "L2",  # Choose L2 or cosine distance based on your preference
        "params": {"nprobe": 10}  # Adjust nprobe for performance
    }
    results = collection.search(
        data=[user_features],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None
    )
    return results

# Step 7: Display Images
def display_similar_images(user_image_path, similar_results, filenames):
    user_img = cv2.imread(user_image_path)
    user_img = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 4, 1)
    plt.imshow(user_img)
    plt.title("User Image")
    plt.axis('off')

    for i, result in enumerate(similar_results[0]):
        img_id = result.id
        if img_id < len(filenames):
            img_filename = filenames[img_id]  # Get the actual filename
            img_path = os.path.join(image_folder, img_filename)  # Construct the path using the filename
            similar_img = cv2.imread(img_path)
            similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)

            plt.subplot(2, 4, i + 2)
            plt.imshow(similar_img)
            plt.title(f"Match {i + 1}\nDistance: {result.distance:.4f} (Lower is better)")
            plt.axis('off')
        else:
            print(f"Image with ID {img_id} not found in the filenames list.")

    plt.tight_layout()
    plt.show()

# Step 8: Main Execution
if __name__ == "__main__":
    image_folder = r'C:\Users\hp\Desktop\temp-folder\test_folder'  # Replace with your images directory
    filenames = insert_features(image_folder)  # Step 5: Insert features into Milvus and get filenames

    user_image_path = r'C:\Users\hp\Desktop\temp-folder\extracted_image_12.jpg'  # Replace with the user image filename
    similar_results = find_similar_images(user_image_path)  # Step 6: Find similar images
    display_similar_images(user_image_path, similar_results, filenames)  # Step 7: Display images