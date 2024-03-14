-- Create the database
CREATE DATABASE mydatabase;

-- Use the database
USE mydatabase;

-- Create the users table
CREATE TABLE users (
   id INT PRIMARY KEY IDENTITY(1,1),
   username VARCHAR(255) NOT NULL,
   password VARCHAR(255) NOT NULL
);
