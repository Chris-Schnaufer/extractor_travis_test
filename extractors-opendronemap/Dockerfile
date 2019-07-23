#FROM clowder/extractors-opendronemap
FROM chrisatua/extractors:opendronemap

MAINTAINER Chris Schnaufer <chris.schnaufer@email.arizona.edu>

# Install any users
RUN id -u extractor \
    || (useradd -u 49044 extractor \
    && mkdir /home/extractor \
    && mkdir /home/extractor/sites \
    && chown -R extractor /home/extractor \
    && chgrp -R extractor /home/extractor)

RUN pip install --upgrade pip && \
    pip install -U numpy && \
    pip install -U utm influxdb laspy && \
    pip install opencv-python-headless && \
    pip install enum34 terrautils

# Copy our source files
COPY entrypoint.sh extractor_info.json *.py /home/extractor/

USER extractor
ENTRYPOINT ["/home/extractor/entrypoint.sh"]
CMD ["extractor"]

# Setup environment variables. These are passed into the container. You can change
# these to your setup. If RABBITMQ_URI is not set, it will try and use the rabbitmq
# server that is linked into the container. MAIN_SCRIPT is set to the script to be
# executed by entrypoint.sh
ENV RABBITMQ_EXCHANGE="terra" \
    RABBITMQ_VHOST="%2F" \
    RABBITMQ_QUEUE="terra.dronepipeline.opendronemap" \
    MAIN_SCRIPT="terraref-opendronemap.py" \
    PYTHONPATH="${PYTHONPATH}:/code"
