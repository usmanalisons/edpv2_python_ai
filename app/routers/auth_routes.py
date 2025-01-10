# app/routers/document_search_routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Query
from pydantic import BaseModel
from app.logic.document_search_logic import DocumentSearchLogic
from app.logic.auth_logic import AuthLogic

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

auth_logic = AuthLogic()

@router.post("/login")
async def login(request: dict = Body(...)):
    try:
        email = request.get("email")
        response = auth_logic.get_employee_by_email(email)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))