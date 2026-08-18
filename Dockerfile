# 1. Pull an official Python base image
FROM python:3.9

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy your requirements list into the container
COPY ./requirements.txt /code/requirements.txt

# 4. Instruct the server to install your Python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy the rest of your repository's files into the container
COPY . .

# 6. Expose port 7860 (Hugging Face's mandatory port)
EXPOSE 7860

# 7. Command the server to run your main routing file
CMD ["python", "main.py"]
