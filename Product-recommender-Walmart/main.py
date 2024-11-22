from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
import pandas as pd

app = FastAPI()
df = pd.read_csv('WMT.csv')

def recommend_similar_products(product_name, data, top_n = 5):
    # Find the category of the selected product
    product_info = data[data['PRODUCT_NAME'].str.lower() == product_name.lower()]

    if product_info.empty:
        return f"Product '{product_name}' not found in the dataset."

    category = product_info['CATEGORY'].iloc[0]
    product_id = product_info['index'].iloc[0]

    # Filter products in the same category
    similar_products = data[(data['CATEGORY'] == category) & (data['index'] != product_id)]

    # Sort by popularity (if purchase count or rating column exists)
    if 'purchase_count' in data.columns:  # Replace with your popularity metric
        similar_products = similar_products.sort_values('purchase_count', ascending=False)

    # Select the top N products
    recommendations = similar_products.drop_duplicates().head(top_n)

    return recommendations

class ProductRequest(BaseModel):
    product_name: str

@app.post("/recommend-products/")
def get_recommendations(product_request: ProductRequest):
    """
    Get product recommendations based on the input product name.

    Args:
        product_name (str): The name of the product to get recommendations for.

    Returns:
        dict: Response containing the recommended products.
    """
    recommendations = recommend_similar_products(product_request.product_name, df)

    return recommendations.to_dict(orient='records')

@app.get("/get_product")
def get_product(no_of_product: str):
    """
    Get product recommendations based on the input product name.

    Args:
        product_name (str): The name of the product to get recommendations for.

    Returns:
        dict: Response containing the recommended products.
    """
    return df.head(int(no_of_product)).to_dict(orient='records')

@app.get("/get_product_by_name")
def get_product_by_name(product_name: str):
    """
    Get product recommendations based on the input product name.

    Args:
        product_name (str): The name of the product to get recommendations for.

    Returns:
        dict: Response containing the recommended products.
    """
    return df[df['PRODUCT_NAME'].str.lower() == product_name.lower()].to_dict(orient='records')

# if __n

