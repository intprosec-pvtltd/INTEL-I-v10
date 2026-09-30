FROM node:24-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
ARG VITE_BASE_URL
ARG VITE_WEBSOCKET_URL
ENV VITE_BASE_URL=$VITE_BASE_URL VITE_WEBSOCKET_URL=$VITE_WEBSOCKET_URL
RUN npm run build
FROM nginxinc/nginx-unprivileged:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
