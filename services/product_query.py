from typing import List

import mysql.connector
from mysql.connector import Error

from models.product_record import ProductRecord

db_config = {
    "host": "uat-mysql.box.pxec.com.tw",
    "user": "roody_wang",
    "password": "DONwF-4SBXRaE4fVetsmK_njQkT@pc86",
    "database": "product",
    "port": 3308
}

def fetch_products() -> List[ProductRecord]:

    sql = """
SELECT p.id AS product_id, p.sale_code, pan.alias_name, pp.pic
FROM product.products p
     JOIN product.product_alias_names pan
          ON p.sale_code = pan.sale_code
     JOIN product.product_pics pp
          ON pp.id = (SELECT MIN(id)
                      FROM product.product_pics
                      WHERE product_id = p.id)
          """

    results: List[ProductRecord] = []
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        for row in cursor:
            rec = ProductRecord(
                product_id=row.get("product_id"),
                sale_code=row.get("sale_code"),
                alias_name=row.get("alias_name"),
                pic=row.get("pic")
            )
            results.append(rec)
        cursor.close()
    except Error as e:
        # 簡單錯誤處理，實作可改為記錄或重新拋出
        print("DB error:", e)
    finally:
        if conn and conn.is_connected():
            conn.close()
    return results


def fetch_product_by_id(product_id: int) -> ProductRecord | None:
    """
    根據 product_id 撈出一筆 ProductRecord，找不到或發生錯誤則回傳 None。
    """

    sql = """
SELECT p.id AS product_id, p.sale_code, pan.alias_name, pp.pic
FROM product.products p
     JOIN product.product_alias_names pan
          ON p.sale_code = pan.sale_code
     JOIN product.product_pics pp
          ON pp.id = (SELECT MIN(id)
                      FROM product.product_pics
                      WHERE product_id = p.id)
WHERE p.id = %s
    """

    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, (product_id,))
        row = cursor.fetchone()
        cursor.close()
        if row:
            return ProductRecord(
                product_id=row.get("product_id"),
                sale_code=row.get("sale_code"),
                alias_name=row.get("alias_name"),
                pic=row.get("pic")
            )
    except Error as e:
        print("DB error:", e)
    finally:
        if conn and conn.is_connected():
            conn.close()
    return None


# 範例使用
if __name__ == "__main__":
    products = fetch_products()
    for p in products:
        print(p)