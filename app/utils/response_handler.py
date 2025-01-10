# app/utils/response_handler.py
def format_search_response(results):
    response = []
    for hits in results:
        for hit in hits:
            item = {
                "score": hit.score,
                "text": hit.entity.get('text'),
                "metadata": hit.entity.get('metadata')
            }
            response.append(item)
    return response
