FROM python:3.11

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/house-pricing-api

#ARG PIP_EXTRA_INDEX_URL

# Install requirements, including from Gemfury
ADD ./house_pricing_api /opt/house-pricing-api
RUN pip install --upgrade pip
RUN pip install -r /opt/house-pricing-api/requirements.txt

RUN chmod +x /opt/house-pricing-api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

EXPOSE 8001

CMD [ "bash", "./run.sh" ]