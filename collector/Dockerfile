FROM php:8.3.4-apache

# Enable Apache modules
RUN a2enmod rewrite headers expires

# Install PHP extensions
RUN docker-php-ext-install pdo pdo_mysql

# PHP upload limits
RUN echo "upload_max_filesize = 10M\npost_max_size = 12M\nmax_execution_time = 60\nmemory_limit = 64M\ndisplay_errors = On\nerror_reporting = E_ALL\nlog_errors = On" \
    > /usr/local/etc/php/conf.d/uploads.ini

# Apache: allow .htaccess overrides
RUN sed -i 's/AllowOverride None/AllowOverride All/g' /etc/apache2/apache2.conf

# Copy app
COPY . /var/www/html/

# Create upload directories
RUN mkdir -p /var/www/html/uploads/pending /var/www/html/uploads/approved \
    && chown -R www-data:www-data /var/www/html/uploads

USER www-data
EXPOSE 80
