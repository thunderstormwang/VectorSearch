# VectorSearch

安裝 python 套件
```
pip install -r requirements.txt
```

啟動 Milvus 服務
```
docker-compose up -d
```

僅停止 Container(保留資源)
```
docker-compose stop
```

停止並移除所有資源(最常用)
停止並刪除所有內容(包含資源)
```
docker-compose down
docker-compose down -v
```

啟動 Attu 服務
```
docker run -d --name attu -p 8192:3000 -e MILVUS_URL=http://host.docker.internal:19530 zilliz/attu:latest
docker run -d --name attu -p 8192:3000 zilliz/attu:latest
```