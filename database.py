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

    def insert_data(self, val):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect_to_database()

            cursor = self.connection.cursor()

            sql = "INSERT INTO goods_statistic (crossed_datetime) VALUES (%s)"
            cursor.execute(sql, (val,))
            self.connection.commit()
            print("Data inserted successfully!")

        except mysql.connector.Error as err:
            print(f"Error: {err}")

        finally:
            if cursor:
                cursor.close()

# import mysql.connector
#
#
# def insert_data(val):
#     try:
#         mydb = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             password="2003",
#             database="warehouse"
#         )
#
#         mycursor = mydb.cursor()
#
#         sql = "INSERT INTO goods_statistic (crossed_datetime) VALUES (%s)"
#         mycursor.execute(sql, (val,))
#         mydb.commit()
#         print("Data inserted successfully!")
#
#     except mysql.connector.Error as err:
#         print(f"Error: {err}")
#
#     finally:
#         if mydb.is_connected():
#             mycursor.close()
#             mydb.close()
