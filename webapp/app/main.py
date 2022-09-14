from fastapi import FastAPI, Depends, HTTPException
from enum import Enum
from typing import Optional
from pydantic import BaseModel

from app.Codigo.opcion_europea_bs import opcion_europea_bs
from app.Codigo.opcion_europea_bin import opcion_europea_bin
from app.Codigo.opcion_europea_mc import opcion_europea_mc
from app.Codigo.opcion_europea_fd import opcion_europea_fd
from app.Codigo.opcion_americana_bin import opcion_americana_bin
from app.Codigo.opcion_americana_fd import opcion_americana_fd

app = FastAPI()

class Model(str, Enum):
    BS = 'BS'
    BIN = 'BIN'
    MC = 'MC'
    FD = 'FD'

class OptionParameters(BaseModel):
    type: str = 'C'
    S: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: Optional[float] = 0.01
    sigma: Optional[float] = 0.1
    div: Optional[float] = 0.0
    model: Model = Model.BS

class Result(BaseModel):
    price:float
    model: Model

@app.get("/price/options/european", tags=['Europeans'])
async def european(params: OptionParameters = Depends()):

    if params.model == Model.BS:
        price = opcion_europea_bs(tipo=params.type,
                                  S=params.S,
                                  K=params.K,
                                  T=params.T,
                                  r=params.r,
                                  sigma=params.sigma,
                                  div=params.div)
    elif params.model == Model.BIN:
         price = opcion_europea_bin(tipo=params.type,
                                  S=params.S,
                                  K=params.K,
                                  T=params.T,
                                  r=params.r,
                                  sigma=params.sigma,
                                  div=params.div,
                                  pasos=1000)
    elif params.model == Model.MC:
         price = opcion_europea_mc(tipo=params.type,
                                  S=params.S,
                                  K=params.K,
                                  T=params.T,
                                  r=params.r,
                                  sigma=params.sigma,
                                  div=params.div,
                                  pasos=1000000)
    elif params.model == Model.FD:
         price = opcion_europea_fd(tipo=params.type,
                                  S=params.S,
                                  K=params.K,
                                  T=params.T,
                                  r=params.r,
                                  sigma=params.sigma,
                                  div=params.div,
                                  M=200)
    else:
        raise HTTPException(status_code=404, detail="Not found")


    return Result(price=price, model=params.model)


@app.get("/price/options/american", tags=['Americans'])
async def american(params: OptionParameters = Depends()):

    if params.model == Model.BIN:
         price = opcion_americana_bin(tipo=params.type,
                                  S=params.S,
                                  K=params.K,
                                  T=params.T,
                                  r=params.r,
                                  sigma=params.sigma,
                                  div=params.div,
                                  pasos=1000)
    elif params.model == Model.FD:
         price = opcion_americana_fd(tipo=params.type,
                                  S=params.S,
                                  K=params.K,
                                  T=params.T,
                                  r=params.r,
                                  sigma=params.sigma,
                                  div=params.div,
                                  M=200)
    else:
        raise HTTPException(status_code=404, detail="Not Implemented")

    return Result(price=price, model=params.model)

class VolatilityParameters(OptionParameters):
    market_price: float = 100
    


@app.get("/IV/american", tags=['IV'])
async def iv_american(params: VolatilityParameters = Depends()):
    ...

@app.get("/IV/european", tags=['IV'])
async def iv_european(params: VolatilityParameters = Depends()):
    ...
