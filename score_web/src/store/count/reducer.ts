import { IInitCountState, IAction } from "./types"
import { ADD, SUB } from "./constants"

const initState: IInitCountState = {
    count: 0
}

const countReducer = (state = initState, action: IAction): IInitCountState => {
    switch(action.type) {
        case ADD: {
            return {...state, count: state.count + action.count}
        }
        case SUB: {
            return {...state, count: state.count - action.count}
        }
        default: {
            return { ...state }
        }
    }
}

export default countReducer