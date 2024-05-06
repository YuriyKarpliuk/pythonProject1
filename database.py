import mysql.connector


class DatabaseHandler:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect_to_database(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print("Connected to database successfully!")
        except mysql.connector.Error as err:
            print(f"Error: {err}")

    def close_connection(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Connection closed.")

    def insert_data(self, datetime, image):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect_to_database()

            cursor = self.connection.cursor()

            sql = "INSERT INTO goods_statistic (crossed_datetime, image_result) VALUES (%s, %s)"
            cursor.execute(sql, (datetime, image,))
            self.connection.commit()
            print("Data inserted successfully!")

        except mysql.connector.Error as err:
            print(f"Error: {err}")

        finally:
            if cursor:
                cursor.close()

    def read_image_by_datetime(self, datetime):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect_to_database()

            cursor = self.connection.cursor()

            sql = "SELECT image_result FROM goods_statistic WHERE crossed_datetime = %s"
            cursor.execute(sql, (datetime,))
            image_data = cursor.fetchone()[0]

            return image_data

        except mysql.connector.Error as err:
            print(f"Error: {err}")

        finally:
            if cursor:
                cursor.close()

    def read_all_datetime_records(self):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect_to_database()

            cursor = self.connection.cursor()

            sql = "SELECT crossed_datetime FROM goods_statistic"
            cursor.execute(sql)
            image_data_list = cursor.fetchall()

            return image_data_list

        except mysql.connector.Error as err:
            print(f"Error: {err}")

        finally:
            if cursor:
                cursor.close()
