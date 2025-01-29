import tiktoken
from app.core.constants import MAX_TOKENS_PER_REQUEST
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, Union
import logging

class Helper:
    @staticmethod
    def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    @staticmethod
    def count_embeddings_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    @staticmethod
    def prepare_policies_procedures_metadatas(chunks, metadatas):
        formatted_metadatas = []
        for chunk, metadata in zip(chunks, metadatas):
            token_count = Helper.count_embeddings_tokens(chunk)
            formatted_metadatas.append({
                "document_id": metadata.get("document_id", ""),
                "access_company": metadata.get("access_company", ""),
                "access_department": metadata.get("access_department", ""),
                "access_employee_emails": metadata.get("access_employee_emails", ""),
                "document_title": metadata.get("document_title", ""),
                "page_number": metadata.get("page_number", 0),
                "chunk_number": metadata.get("chunk_number", 0),
                "token_count": token_count,
            })
        return formatted_metadatas

    @staticmethod
    def prepare_oracle_trainings_metadatas(chunks, metadatas):
        formatted_metadatas = []
        for chunk, metadata in zip(chunks, metadatas):
            token_count = Helper.count_embeddings_tokens(chunk)
            formatted_metadatas.append({
                "document_id": metadata.get("id", ""),
                "category_id": metadata.get("categoryId", ""),
                "document_title": metadata.get("title", ""),
                "page_number": metadata.get("page_number", 0),
                "chunk_number": metadata.get("chunk_number", 0),
                "token_count": token_count,
            })
        return formatted_metadatas

    @staticmethod
    def batch_process(chunks, metadatas, max_tokens=MAX_TOKENS_PER_REQUEST):
        batches = []
        batch_chunks = []
        batch_metadatas = []
        batch_total_tokens = 0

        for chunk, metadata in zip(chunks, metadatas):
            chunk_token_count = metadata.get("token_count", 0)
            if batch_total_tokens + chunk_token_count > max_tokens:
                print('TOKEN LIMIT EXCEEDS, CREATING EMBEDDINGS...')
                batches.append((batch_chunks, batch_metadatas))
                batch_chunks, batch_metadatas, batch_total_tokens = [], [], 0

            batch_chunks.append(chunk)
            batch_metadatas.append(metadata)
            batch_total_tokens += chunk_token_count

        if batch_chunks:
            batches.append((batch_chunks, batch_metadatas))

        return batches
    
    # @staticmethod
    # def filter_policies_procedures(self, email: str, department_code: str, company_code: str,):
    #     conditions = [
    #         '(access_company == "All" and access_department == "All")',
    #         f'(access_employee_emails LIKE "%{email}%")',
    #         f'(access_company == "All" and access_department LIKE "%{department_code}%")',
    #         f'(access_company LIKE "%{company_code}%" and access_department == "All")',
    #         f'(access_company LIKE "%{company_code}%" and access_department LIKE "%{department_code}%")',
    #     ]
    #     return " or ".join(conditions)

    @staticmethod
    def filter_policies_procedures(email: str, department_code: str, company_code: str) -> dict:
        filters = {
           "$or": [
            {
                "$and": [
                    {"access_company": {"$eq": "All"}},
                    {"access_department": {"$eq": "All"}}
                ]
            },
            {"access_employee_emails": {"$in": [email]}},
            {
                "$and": [
                    {"access_company": {"$eq": "All"}},
                    {"access_department": {"$in": [department_code]}}
                ]
            },
            {
                "$and": [
                    {"access_company": {"$in": [company_code]}},
                    {"access_department": {"$eq": "All"}}
                ]
            },
            {
                "$and": [
                    {"access_company": {"$in": [company_code]}},
                    {"access_department": {"$in": [department_code]}}
                ]
            }
        ]
        }
        return filters

    
    # @staticmethod
    # def plot_chart(rows, chart_type="bar_chart", output_dir="app/charts", chart_name_prefix="chart"):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     if not rows or not isinstance(rows, list) or len(rows) == 0:
    #         return "No data available for chart."

    #     x_col = list(rows[0].keys())[0]
    #     y_col = list(rows[0].keys())[1]

    #     valid_rows = [
    #         row for row in rows
    #         if isinstance(row.get(x_col), (int, float, str))
    #         and isinstance(row.get(y_col), (int, float))
    #     ]

    #     if not valid_rows:
    #         return "No valid data available for chart."

    #     x_vals = [row[x_col] for row in valid_rows]
    #     y_vals = [row[y_col] for row in valid_rows]

    #     fig, ax = plt.subplots(figsize=(6, 4))

    #     if chart_type == "bar_chart":
    #         ax.bar(x_vals, y_vals)  # Vertical bars for bar chart
    #     elif chart_type == "column_chart":
    #         ax.bar(x_vals, y_vals)  # Column chart is essentially the same as bar chart in this context
    #     elif chart_type == "line_chart":
    #         ax.plot(x_vals, y_vals, marker='o', color='blue', linestyle='-', linewidth=2)  # Line chart customization
    #     elif chart_type == "pie_chart":
    #         ax.pie(y_vals, labels=x_vals, autopct='%1.1f%%')
    #     else:
    #         return "Unsupported chart type."

    #     ax.set_xlabel(x_col)
    #     ax.set_ylabel(y_col)

    #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    #     filename = f"{chart_name_prefix}_{timestamp}.png"
    #     filepath = os.path.join(output_dir, filename)

    #     plt.savefig(filepath)
    #     plt.close(fig)

    #     return filepath

    @staticmethod
    def plot_chart(rows, chart_type="bar_chart", output_dir="app/charts", chart_name_prefix="chart"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not rows or not isinstance(rows, list) or len(rows) == 0:
            return "No data available for chart."

        try:
            x_vals = [str(row["x_col"]) for row in rows]
            y_vals = [float(row["y_col"]) for row in rows]

            plt.figure(figsize=(8, 6), dpi=100)  # Set a default figure size for all charts
            
            if chart_type == "bar_chart":
                plt.bar(x_vals, y_vals)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Values")
                plt.xlabel("Categories")
                plt.title("Bar Chart")

            elif chart_type == "line_chart":
                plt.plot(x_vals, y_vals, marker='o')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Values")
                plt.xlabel("Categories")
                plt.title("Line Chart")

            elif chart_type == "pie_chart":
                # Increase size and adjust label spacing for better readability
                plt.figure(figsize=(8, 6), dpi=100)
                plt.pie(
                    y_vals, 
                    labels=x_vals, 
                    autopct='%1.1f%%', 
                    pctdistance=0.85, 
                    labeldistance=1.1
                )
                plt.title("Pie Chart")
                plt.tight_layout()

            else:
                return "Unsupported chart type."

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{chart_name_prefix}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)

            # Save the chart with specified DPI and padding
            plt.savefig(filepath, bbox_inches='tight', dpi=100, pad_inches=0.2)
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error plotting chart: {str(e)}")
            return "Could not generate the graph"
        

    @staticmethod
    def generate_chart(
        chart_data: List[Dict[str, Union[str, float, Decimal]]],
        chart_type: str = "bar_chart",
        chart_name: str = "chart",
        output_dir: str = "app/charts"
    ) -> Optional[str]:
        """
        Generates a visualization chart from the provided data and saves it as a PNG file.
        
        This method handles the generation of three types of charts:
        1. Bar charts: For comparing categorical data
        2. Line charts: For showing trends over time or sequences
        3. Pie charts: For showing proportions of a whole

        Args:
            chart_data: List of dictionaries containing x_col and y_col values
                    Example: [{"x_col": "Category A", "y_col": 100.5}, ...]
            chart_type: Type of chart to generate ("bar_chart", "line_chart", or "pie_chart")
            chart_name: Base name for the output file (timestamp will be appended)
            output_dir: Directory where the chart should be saved

        Returns:
            str: Path to the generated chart file, or None if generation fails
        
        Example:
            data = [{"x_col": "Jan", "y_col": 1000}, {"x_col": "Feb", "y_col": 1200}]
            filepath = Helper.generate_chart(data, "line_chart", "monthly_sales")
        """
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Validate input data
        if not chart_data or not isinstance(chart_data, list):
            logging.error("Invalid or empty chart data provided")
            return None

        try:
            # Process and prepare the data
            formatted_data = []
            for row in chart_data:
                # Validate required columns
                if "x_col" not in row or "y_col" not in row:
                    continue
                    
                # Convert y-values to float and x-values to string
                y_val = float(row["y_col"]) if isinstance(row["y_col"], (int, float, Decimal)) else None
                x_val = str(row["x_col"])
                
                if y_val is not None:
                    formatted_data.append({"x_col": x_val, "y_col": y_val})

            # Handle empty dataset after formatting
            if not formatted_data:
                logging.warning("No valid data points after formatting")
                return None

            # Sort data based on chart type
            if chart_type == "pie_chart":
                # Sort by value descending and limit to top 8 categories
                formatted_data.sort(key=lambda x: x["y_col"], reverse=True)
                if len(formatted_data) > 8:
                    other_sum = sum(row["y_col"] for row in formatted_data[8:])
                    formatted_data = formatted_data[:8]
                    formatted_data.append({"x_col": "Others", "y_col": other_sum})
            elif chart_type == "line_chart":
                # Sort by x-value for line charts
                formatted_data.sort(key=lambda x: x["x_col"])

            # Extract x and y values
            x_vals = [row["x_col"] for row in formatted_data]
            y_vals = [row["y_col"] for row in formatted_data]

            # Create the figure with appropriate size and resolution
            plt.figure(figsize=(10, 6), dpi=100)

            # Generate the appropriate chart type
            if chart_type == "bar_chart":
                plt.bar(x_vals, y_vals, color='skyblue')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Value")
                plt.xlabel("Category")
                # Add value labels on top of bars
                for i, v in enumerate(y_vals):
                    plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')

            elif chart_type == "line_chart":
                plt.plot(x_vals, y_vals, marker='o', color='skyblue', linewidth=2)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Value")
                plt.xlabel("Time Period")
                # Add value labels at each point
                for i, v in enumerate(y_vals):
                    plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')

            elif chart_type == "pie_chart":
                plt.pie(
                    y_vals,
                    labels=x_vals,
                    autopct='%1.1f%%',
                    pctdistance=0.85,
                    labeldistance=1.1,
                    startangle=90
                )
                plt.axis('equal')
            else:
                logging.error(f"Unsupported chart type: {chart_type}")
                plt.close()
                return None

            # Add title and adjust layout
            chart_titles = {
                "bar_chart": "Categorical Comparison",
                "line_chart": "Trend Analysis",
                "pie_chart": "Distribution Analysis"
            }
            plt.title(chart_titles.get(chart_type, "Data Visualization"))
            plt.tight_layout(pad=1.5)

            # Generate unique filename and save
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{chart_name}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)

            # Save with high quality settings
            plt.savefig(
                filepath,
                bbox_inches='tight',
                dpi=100,
                pad_inches=0.2
            )
            plt.close()

            return filepath

        except Exception as e:
            logging.error(f"Error generating chart: {str(e)}")
            if plt.get_fignums():
                plt.close()
            return None
        

    @staticmethod
    def convert_decimals_in_obj(obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, list):
            return [Helper.convert_decimals_in_obj(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: Helper.convert_decimals_in_obj(value) for key, value in obj.items()}
        return obj



