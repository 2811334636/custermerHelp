-- ch02 · 测试库
-- mysql 官方镜像的 entrypoint 只会把 app 用户授权到 MYSQL_DATABASE(customer_service)，
-- 不会授权测试库。而 tests/ 用 app 连 customer_service_test，seed.py 也要在那里建表，
-- 所以这里显式建库 + 授权。放在 01-schema.sql 之后按文件名序执行。
--
-- 注意：docker-entrypoint-initdb.d 只在数据卷首次初始化时执行一次。
-- 改本文件后需要 `docker compose down -v && docker compose up -d` 才生效。

CREATE DATABASE IF NOT EXISTS customer_service_test
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

GRANT ALL PRIVILEGES ON customer_service_test.* TO 'app'@'%';
FLUSH PRIVILEGES;
