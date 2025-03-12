FROM node:20

#ENV NODE_OPTIONS=--openssl-legacy-provider

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . . 

EXPOSE 3000

# CMD ["npm", "run", "build"]

CMD ["npm", "start"]