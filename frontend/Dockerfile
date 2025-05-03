# ✅ 前端容器（构建 React 并用 Nginx 托管）
FROM node:18-alpine as build

WORKDIR /app
COPY . .

RUN npm install && npm run build

# 用 Nginx 托管静态资源
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80