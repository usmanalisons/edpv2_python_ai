from app.services.sql_db_service import SQLDatabaseService

class AuthLogic:

    def get_employee_by_email(self, email):
        sql_service = SQLDatabaseService(connection_name="intranet")
        employee = sql_service.get_employee_by_email(email)
        return {
            "data": employee 
        }
    