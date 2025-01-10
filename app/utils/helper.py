import tiktoken
from app.core.constants import MAX_TOKENS_PER_REQUEST
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Helper:
    @staticmethod
    def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    @staticmethod
    def prepare_policies_procedures_metadatas(chunks, metadatas):
        formatted_metadatas = []
        for chunk, metadata in zip(chunks, metadatas):
            token_count = Helper.count_tokens(chunk)
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
            token_count = Helper.count_tokens(chunk)
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
        """
        Generate chart from formatted data with fixed dimensions of 600x400 pixels.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not rows or not isinstance(rows, list) or len(rows) == 0:
            return "No data available for chart."

        try:
            x_vals = [str(row["x_col"]) for row in rows]
            y_vals = [float(row["y_col"]) for row in rows]

            # Create figure with fixed size in inches (600x400 pixels at 100 DPI)
            plt.figure(figsize=(6, 4), dpi=100)  # 6x4 inches at 100 DPI = 600x400 pixels
            
            if chart_type == "bar_chart":
                plt.bar(x_vals, y_vals)
                plt.xticks(rotation=45, ha='right')
            elif chart_type == "line_chart":
                plt.plot(x_vals, y_vals, marker='o')
                plt.xticks(rotation=45, ha='right')
            elif chart_type == "pie_chart":
                # For pie charts, use a slightly different aspect ratio to maintain circular shape
                plt.figure(figsize=(5, 4), dpi=100)  # Adjusted for pie chart
                plt.pie(y_vals, labels=x_vals, autopct='%1.1f%%')
            else:
                return "Unsupported chart type."

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{chart_name_prefix}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)

            # Save with fixed DPI and size
            plt.savefig(filepath, 
                       bbox_inches='tight',
                       dpi=100,  # Set DPI to 100 for direct pixel mapping
                       pad_inches=0.2)  # Add small padding
            plt.close()

            return filepath

        except Exception as e:
            print(f"Error plotting chart: {str(e)}")
            return "Error generating chart."



