# ---- build: compile and package into a single runnable jar
FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /build
COPY pom.xml .
RUN mvn -q -B dependency:go-offline
COPY src ./src
RUN mvn -q -B package

# ---- run
FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /build/target/bm25-lab.jar /app/bm25-lab.jar
ENTRYPOINT ["java", "-jar", "/app/bm25-lab.jar"]
CMD ["benchmark"]
