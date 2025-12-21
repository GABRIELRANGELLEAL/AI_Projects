# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Ingest

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary
# MAGIC
# MAGIC This notebook will be used to ingest the data, how it works:
# MAGIC - Upload the files in the dbfs 
# MAGIC - Create three schmeas 
# MAGIC   - Bronze = raw_data
# MAGIC   - silver = store_data
# MAGIC   - gold = summary_data prepared for the dash

# COMMAND ----------

from pyspark.sql.utils import AnalysisException

# COMMAND ----------

# Creating schema 
spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")


# checking if the files was correctly uploaded to dbfs
print('Below there a check file')
for item in dbutils.fs.ls("/FileStore/tables/"):
  print(item)

# reading csv and storing this into some variables
print('-' * 100)
print('Below there a check file')
products = spark.read.option("header", True).csv("/FileStore/tables/products.csv")
sales_detail = spark.read.option("header", True).csv("/FileStore/tables/sales_order_detail.csv")
sales_header= spark.read.option("header", True).csv("/FileStore/tables/sales_order_header.csv")

# printing a sample of dataframes
products.sample(fraction = 0.25).display()
sales_detail.sample(fraction = 0.25).display()
sales_header.sample(fraction = 0.25).display()

# COMMAND ----------

# DBTITLE 1,creating tables in our schema
# creating the tables in the database
try:
  product_table = spark.table("bronze.raw_products")
  if product_table.rdd.isEmpty():
    products.write.mode("overwrite").saveAsTable("bronze.raw_products")
  else:
    print("Exisitng table contain data, overwrite manually if wan't to procced.")
except AnalysisException:
  products.write.mode("overwrite").saveAsTable("bronze.raw_products")

try:
  sales_detail_table = spark.table("bronze.raw_sales_detail")
  if sales_detail_table.rdd.isEmpty():
    sales_detail.write.mode("overwrite").saveAsTable("bronze.raw_sales_detail")
  else:
    print("Exisitng table contain data, overwrite manually if wan't to procced.")
except AnalysisException:
  sales_detail.write.mode("overwrite").saveAsTable("bronze.raw_sales_detail")

try:
  sales_header_table = spark.table("bronze.raw_sales_header")
  if sales_header_table.rdd.isEmpty():
    sales_header.write.mode("overwrite").saveAsTable("bronze.raw_sales_header")
  else:
    print("Exisitng table contain data, overwrite manually if wan't to procced.")
except AnalysisException:
  sales_header.write.mode("overwrite").saveAsTable("bronze.raw_sales_header")

