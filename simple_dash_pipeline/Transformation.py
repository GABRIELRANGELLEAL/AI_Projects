# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Transformations

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary
# MAGIC
# MAGIC This notebook will be used to transform the data, how it works:
# MAGIC - assign appropriate data types
# MAGIC - Replace NULL values in the Color field with N/A.
# MAGIC - Enhance the ProductCategoryName field when it is NULL using the following logic:
# MAGIC   - If ProductSubCategoryName is in (‘Gloves’, ‘Shorts’, ‘Socks’,‘Tights’, ‘Vests’), set ProductCategoryName to ‘Clothing’.
# MAGIC   - If ProductSubCategoryName is in (‘Locks’, ‘Lights’, ‘Headsets’,‘Helmets’, ‘Pedals’, ‘Pumps’), set ProductCategoryName to ‘Accessories’.
# MAGIC   - If ProductSubCategoryName contains the word ‘Frames’ or is in (‘Wheels’, ‘Saddles’), set ProductCategoryName to ‘Components’.

# COMMAND ----------

# DBTITLE 1,imports
from pyspark.sql.functions import round as round_spark, col, when, expr, datediff, transform,dayofweek, size, filter as array_filter
from pyspark.sql.utils import AnalysisException

# COMMAND ----------

# MAGIC %md
# MAGIC #### Assign appropriate data types and replacing null

# COMMAND ----------

# DBTITLE 1,printing data types
raw_products = spark.table("bronze.raw_products")
raw_sales_detail = spark.table("bronze.raw_sales_detail")
raw_sales_header = spark.table("bronze.raw_sales_header")

# COMMAND ----------

store_product = raw_products.selectExpr(
    "cast(ProductID as int) as ProductID",
    "ProductDesc",
    "ProductNumber",
    "cast(MakeFlag as boolean) as MakeFlag",
    "Color",
    "cast(SafetyStockLevel as int) as SafetyStockLevel",
    "cast(ReorderPoint as int) as ReorderPoint",
    "round(cast(StandardCost as double), 2) as StandardCost",
    "round(cast(ListPrice as double), 2) as ListPrice",
    "Size",
    "SizeUnitMeasureCode",
    "round(cast(Weight as double), 2) as Weight",
    "WeightUnitMeasureCode",
    "ProductCategoryName",
    "ProductSubCategoryName"
)
store_sales_detail= raw_sales_detail.selectExpr(
    "cast(SalesOrderID as int) as SalesOrderID",
    "cast(SalesOrderDetailID as int) as SalesOrderDetailID",
    "cast(OrderQty as int) as OrderQty",
    "cast(ProductID as int) as ProductID",
    "round(cast(UnitPrice as double), 2) as UnitPrice",
    "round(cast(UnitPriceDiscount as double), 2) as UnitPriceDiscount"
)
store_sales_header= raw_sales_header.selectExpr(
    "cast(SalesOrderID as int) as SalesOrderID",
    "cast(OrderDate as date) as OrderDate",
    "cast(ShipDate as date) as ShipDate",
    "cast(OnlineOrderFlag as boolean) as OnlineOrderFlag",
    "AccountNumber",
    "cast(CustomerID as int) as CustomerID",
    "cast(SalesPersonID as int) as SalesPersonID",
    "round(cast(Freight as double), 2) as Freight"
)

store_product.sample(fraction = 0.25).display()
store_sales_detail.sample(fraction = 0.25).display()
store_sales_header.sample(fraction = 0.25).display()

# COMMAND ----------

# spark.sql("DROP TABLE IF EXISTS silver.store_products")
# dbutils.fs.rm("dbfs:/user/hive/warehouse/silver.db/sales_detail_store", recurse=True)

store_product.write.mode("overwrite").saveAsTable("silver.store_products")
store_sales_detail.write.mode("overwrite").saveAsTable("silver.store_sales_detail")
store_sales_header.write.mode("overwrite").saveAsTable("silver.store_sales_header")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Product Master Transformations:
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Perform the following transformations on the **product master** data and write the results into a table named publish_product:
# MAGIC - Replace NULL values in the **Color** field with N/A.
# MAGIC - Enhance the ProductCategoryName field when it is NULL using the following logic:
# MAGIC   - If ProductSubCategoryName is in (‘Gloves’, ‘Shorts’, ‘Socks’,‘Tights’, ‘Vests’), set ProductCategoryName to ‘Clothing’.
# MAGIC   - If ProductSubCategoryName is in (‘Locks’, ‘Lights’, ‘Headsets’,‘Helmets’, ‘Pedals’, ‘Pumps’), set ProductCategoryName to ‘Accessories’.
# MAGIC   - If ProductSubCategoryName contains the word ‘Frames’ or is in (‘Wheels’, ‘Saddles’), set ProductCategoryName to ‘Components’

# COMMAND ----------

publish_product = spark.table("silver.store_products")

# Replace NULL values in the Color field with N/A.
publish_product = publish_product.fillna({'Color': 'N/A'})

# ProductCategoryName field when it is NULL using the following logic
publish_product = (
  publish_product
  .withColumn('ProductCategoryNameEnhaced',
    # If ProductSubCategoryName is in (‘Gloves’, ‘Shorts’, ‘Socks’,‘Tights’, ‘Vests’), set ProductCategoryName to ‘Clothing’.
    when(
      col('ProductCategoryName').isNull()
      &col("ProductSubCategoryName").isin("Gloves", "Shorts", "Socks", "Tights", "Vests"),
        "Clothing"
    )
    # If ProductSubCategoryName is in (‘Locks’, ‘Lights’, ‘Headsets’,‘Helmets’, ‘Pedals’, ‘Pumps’), set ProductCategoryName to ‘Accessories’.
    .when(
        col("ProductCategoryName").isNull() 
        & col("ProductSubCategoryName").isin("Locks", "Lights", "Headsets", "Helmets", "Pedals", "Pumps"),
        "Accessories"
    )
    # If ProductSubCategoryName contains the word ‘Frames’ or is in (‘Wheels’, ‘Saddles’), set ProductCategoryName to ‘Components’
    .when(
        col("ProductCategoryName").isNull() 
        & (
            col("ProductSubCategoryName").like("%Frames%") |
            col("ProductSubCategoryName").isin("Wheels", "Saddles")
        ),
        "Components"
    )
    .otherwise(col("ProductCategoryName"))
  )
)

# overwrite
# spark.sql("DROP TABLE IF EXISTS silver.store_products")
# dbutils.fs.rm("dbfs:/user/hive/warehouse/silver.db/sales_detail_store", recurse=True)
publish_product.write.mode("overwrite").saveAsTable("silver.publish_product")

# COMMAND ----------

# DBTITLE 1,printing the diff between two tables
store_products = spark.table("silver.store_products")
publish_product = spark.table("silver.publish_product")
store_products.display()
publish_product.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sales Order Transformations:

# COMMAND ----------

# MAGIC %md
# MAGIC Join SalesOrderDetail with SalesOrderHeader on SalesOrderId and apply the following transformations:
# MAGIC - Calculate LeadTimeInBusinessDays as the difference between OrderDate and ShipDate, excluding Saturdays and Sundays.
# MAGIC - Calculate TotalLineExtendedPrice using the formula: OrderQty * (UnitPrice - UnitPriceDiscount).
# MAGIC - Write the results into a table named publish_orders, including:
# MAGIC   -All fields from SalesOrderDetail.

# COMMAND ----------

publish_product = spark.table("silver.publish_product")
store_sales_detail = spark.table("silver.store_sales_detail")
store_sales_header = spark.table("silver.store_sales_header")

publish_orders = (
  store_sales_header.alias('ssh')
  .join(store_sales_detail.alias('ssd'), on = 'SalesOrderId', how = 'inner')
  .withColumnRenamed("Freight", "TotalOrderFreight")
  .withColumn(
    "array_date",
    expr('''
        sequence(OrderDate, ShipDate, interval 1 day)
    ''')
    )
  .withColumn(
    "LeadTimeInBusinessDays",
    # array_filter vai iterar cada data 'd' e reter só as que não são sábado(7) nem domingo(1)
    size(
        array_filter(
            col("array_date"),
            lambda d: (dayofweek(d) != 1) & (dayofweek(d) != 7)
        )
    )
  )
  .withColumn(
    "TotalLineExtendedPrice",
    col("OrderQty") * (col("UnitPrice") - col("UnitPriceDiscount"))
  )
  .drop('ssh.SalesOrderID', 'array_date')
)

publish_orders.display()

# COMMAND ----------

# overwrite
# spark.sql("DROP TABLE IF EXISTS silver.publish_orders")
# dbutils.fs.rm("dbfs:/user/hive/warehouse/silver.db/publish_orders", recurse=True)
publish_orders.write.mode("overwrite").saveAsTable("silver.publish_orders")
spark.table("silver.publish_orders").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis Questions:

# COMMAND ----------

# MAGIC %md
# MAGIC Provide answers to the following questions based on the transformed data:
# MAGIC - Which color generated the highest revenue each year?
# MAGIC - What is the average LeadTimeInBusinessDays by ProductCategoryName?

# COMMAND ----------

publish_orders = spark.table("silver.publish_orders")
publish_product= spark.table("silver.publish_product")
master_order_product = (
  publish_orders.alias('po')
  .join(publish_product.alias('pp'), on = 'ProductID', how = 'inner')
  .select(
    'pp.ProductID'
    ,'po.SalesOrderID'
    ,'po.ShipDate'
    ,'pp.Color'
    ,'po.TotalLineExtendedPrice'
    ,'pp.ProductCategoryNameEnhaced'
    ,'po.LeadTimeInBusinessDays'
  )
)
master_order_product.write.mode("overwrite").saveAsTable("gold.master_order_product")
spark.table("gold.master_order_product").display()

# COMMAND ----------

# MAGIC %md 
# MAGIC To answare the questions I have created a very simply Power BI dashboard to access the dash click in this [link](https://app.powerbi.com/view?r=eyJrIjoiMDRlZTc2N2ItOGQ1My00MTdkLWJlMTItMDA0YTdkYTRlNmRkIiwidCI6ImIzNGMxZDU1LWE0M2UtNGEyMC05MjE4LWExYTQyZWFiMTQ5YSJ9)