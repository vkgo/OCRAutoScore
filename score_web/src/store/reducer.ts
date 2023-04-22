//reducer.ts
import { combineReducers } from "redux"
import countReducer from "./count"

import { IInitCountState } from "./count/types"

export interface IRootReducer {   //合并reducer之后，state的类型
    countReducer: IInitCountState
}

const rootReducer =  combineReducers<IRootReducer>({
    countReducer
})

export default rootReducer
