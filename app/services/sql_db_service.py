from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from app.core.config import settings
import logging

class SQLDatabaseService:
    def __init__(self, connection_name="intranet"):
        self.connection_string = settings.SQL_CONNECTION_STRINGS.get(connection_name)
        if not self.connection_string:
            raise ValueError(f"No connection string found for '{connection_name}'")
        
        self.engine = create_engine(self.connection_string, pool_pre_ping=True)
        # self.SessionLocal = sessionmaker(bind=self.engine)
        self.SessionLocal = scoped_session(sessionmaker(bind=self.engine))

    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            logging.error(f"Database session error: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def get_employee_by_email(self, employee_email: str):
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT 
                        Emp_ID as employee_id, 
                        emp_name as employee_name, 
                        email as employee_email, 
                        comp_code as company_code, 
                        company_name, 
                        dept_code as department_code, 
                        department as department_name, 
                        desig_code as designation_code, 
                        designation as designation_name 
                    FROM v_employee_list 
                    WHERE email LIKE :employee_email;
                """)
                
                result = session.execute(query, {"employee_email": f"%{employee_email}%"})
                row = result.fetchone()

                columns = result.keys()
                if not row:
                    logging.info(f"No employee found with email: {employee_email}")
                    return None
                
                row_data = {column: value for column, value in zip(columns, row)}
                return row_data
        except Exception as e:
            logging.error(f"Error retrieving employee by email: {e}")
            return None


    def get_policies_procedures_from_db(self, fileType='pdf'):
        action = 'getAll'
        result_data = []
        try:
            with self.get_session() as session:
                query = text("""
                    EXEC sp_document_embedding_actions 
                        @fileType = :fileType,
                        @action = :action;
                """)
                result = session.execute(query, {"fileType": fileType, "action": action})
                rows = result.fetchall()

                if not rows:
                    raise Exception("Stored procedure did not return any results.")

                columns = result.keys()
                for row in rows:
                    row_data = {column: value for column, value in zip(columns, row)}
                    result_data.append(row_data)
            return result_data
        except Exception as e:
            logging.error(f"Error: {e}")
            return []


    def update_policies_procedures_last_embedding_at(self, document_ids):
        action = 'updateEmbeddingAt'
        try:
            with self.get_session() as session:
                query = text("""
                    EXEC sp_document_embedding_actions
                        @documentIds = :documentIds, 
                        @action = :action;
                """)
                session.execute(query, {"documentIds": document_ids, "action": action})
                session.commit()
                logging.info(f"Successfully updated embedding for document number {str(document_ids)}.")
            return True
        except Exception as e:
            logging.error(f"Error: {e}")
            return False


    def run_query(self, query: str):
        results = []
        session = None
        try:
            logging.info(f"Preparing to execute query: {query}")
            with self.get_session() as session:
                result = session.execute(text(query))
                rows = result.fetchall()

                logging.info(f"Rows fetched: {len(rows) if rows else 0}")

                if not rows:
                    logging.warning("Query executed successfully but returned no results.")
                    return results

                columns = list(result.keys())
                logging.info(f"Query executed successfully. Columns: {columns}")

                for row in rows:
                    row_data = {columns[i]: row[i] for i in range(len(columns))}
                    results.append(row_data)

            return results

        except Exception as e:
            logging.error(f"Error running query: {e}", exc_info=True)
            return []
        finally:
            if session:
                session.close()
                logging.info("Database session closed.")


    def get_table_schema(self, table_name: str):
        try:
            with self.get_session() as session:
                inspector = inspect(self.engine)
                columns = inspector.get_columns(table_name)
                table_comment = inspector.get_table_comment(table_name).get("text", "")

                result_data = []
                for column in columns:
                    result_data.append({
                        "COLUMN_NAME": column['name'],
                        "DATA_TYPE": str(column['type']),
                        "COLUMN_COMMENT": column.get("comment", None),
                        "TABLE_COMMENT": table_comment
                    })
            return result_data
        except Exception as e:
            logging.error(f"Error: {e}")
            return []
        
    def get_oracle_trainings_from_db(self):
        result_data = []
        try:
            with self.get_session() as session:
                query = text("""
                   SELECT * FROM  quantum_learning_manuals WHERE categoryId >= 8;
                """)
                result = session.execute(query)
                rows = result.fetchall()

                if not rows:
                    raise Exception("Stored procedure did not return any results.")

                columns = result.keys()
                for row in rows:
                    row_data = {column: value for column, value in zip(columns, row)}
                    result_data.append(row_data)
            return result_data
        except Exception as e:
            logging.error(f"Error: {e}")
            return []
        

    def get_employee_chats(self, employee_email, chat_type):
        result_data = []
        try:
            with self.get_session() as session:
                query = text("""
                   SELECT * FROM chats WHERE employee_email = :employee_email AND chat_type = :chat_type ORDER BY updated_at DESC;
                """)
                params = {"employee_email": employee_email, "chat_type": chat_type}
                result = session.execute(query, params)
                rows = result.fetchall()
                if not rows:
                   return []

                columns = result.keys()
                for row in rows:
                    row_data = {column: value for column, value in zip(columns, row)}
                    result_data.append(row_data)
            return result_data
        except Exception as e:
            logging.error(f"Error: {e}")
            return []
        
    def get_chat_by_id(self, chat_id: str):
        try:
            with self.get_session() as session:
                query = text("""
                    SELECT * FROM chats WHERE chat_uid = :chat_id;
                """)
                result = session.execute(query, {"chat_id": chat_id})
                row = result.fetchone()

                columns = result.keys()
                if not row:
                    logging.info(f"No chat found with ID: {chat_id}")
                    return None
                row_data = {column: value for column, value in zip(columns, row)}
                return row_data
        except Exception as e:
            logging.error(f"Error retrieving chat by ID: {e}")
            return None



    def add_chat(self, chat_uid: str, chat_type: str, employee_email: str, chat_title: str):
        newChat = None
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO chats (chat_uid, chat_type, employee_email, chat_title, created_at, updated_at)
                    VALUES (:chat_uid, :chat_type, :employee_email, :chat_title, GETDATE(), GETDATE());
                """)
                params = {
                    "chat_uid": chat_uid,
                    "chat_type": chat_type,
                    "employee_email": employee_email,
                    "chat_title": chat_title,
                }
                session.execute(query, params)
                session.commit()

                newChat = self.get_chat_by_id(chat_uid)

                logging.info(f"Chat with UID {chat_uid} and title '{chat_title}' added successfully.")

            return newChat
        except Exception as e:
            logging.error(f"Error adding chat: {e}")
            return newChat


    def add_chat_message(self, chat_uid: str, message_type: str, message_text: str, message_html: str = None, message_json: str = None, sql_query: str = None):
        try:
            with self.get_session() as session:
                query = text("""
                    INSERT INTO chat_messages (chat_uid, message_type, message_text, message_html, message_json, sql_query, created_at, updated_at)
                    VALUES (:chat_uid, :message_type, :message_text, :message_html, :message_json, :sql_query, GETDATE(), GETDATE());
                """)
                session.execute(query, {
                    "chat_uid": chat_uid, 
                    "message_type": message_type, 
                    "message_text": message_text, 
                    "message_html": message_html, 
                    "message_json": message_json,
                    "sql_query": sql_query
                })
                session.commit()
                logging.info(f"Message added to chat {chat_uid} successfully.")
            return True
        except Exception as e:
            logging.error(f"Error adding chat message: {e}")
            return False


    def get_chat_messages(self, chat_uid: str, message_type: str = None):
        result_data = []
        try:
            with self.get_session() as session:
                query = "SELECT * FROM chat_messages WHERE chat_uid = :chat_uid"
                params = {"chat_uid": chat_uid}

                if message_type:
                    query += " AND message_type = :message_type"
                    params["message_type"] = message_type

                result = session.execute(text(query), params)
                rows = result.fetchall()
                columns = result.keys()
                for row in rows:
                    row_data = {column: value for column, value in zip(columns, row)}
                    result_data.append(row_data)
            return result_data
        except Exception as e:
            logging.error(f"Error retrieving chat messages: {e}")
            return []
        

    def get_top_distinct_ctc_data(self):
        result_data = []
        try:
            with self.get_session() as session:
                query = text("""
                   WITH RankedData AS (
                        SELECT 
                            *,
                            ROW_NUMBER() OVER (PARTITION BY COMPANY_CODE ORDER BY COMPANY_CODE) AS RowNum
                        FROM 
                            dbo.VIEW_COMPANY_PROJECT_COST_TO_COMPLETE
                    )
                    SELECT 
                        *
                    FROM 
                        RankedData
                    WHERE 
                        RowNum = 1
                    ORDER BY 
                        COMPANY_CODE
                    OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY;

                """)
                result = session.execute(query)
                rows = result.fetchall()

                if not rows:
                    raise Exception("Stored procedure did not return any results.")

                columns = result.keys()
                for row in rows:
                    row_data = {column: value for column, value in zip(columns, row)}
                    result_data.append(row_data)
            return result_data
        except Exception as e:
            logging.error(f"Error in get_top_distinct_ctc_data: {e}")
            return []
