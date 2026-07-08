import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

import database
import auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

class UserResponse(BaseModel):
    id: int
    username: str

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister):
    """Register a new user."""
    # Check if user already exists
    existing_user = database.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_pwd = auth.hash_password(user_data.password)
    user_id = database.create_user(user_data.username, hashed_pwd)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create user"
        )
    
    return UserResponse(id=user_id, username=user_data.username)

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate a user and return a JWT access token."""
    user = database.get_user_by_username(form_data.username)
    if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(
        data={"sub": user["username"], "id": user["id"]}
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        username=user["username"]
    )

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(auth.get_current_user)):
    """Get the currently logged-in user profile."""
    return UserResponse(id=current_user["id"], username=current_user["username"])
