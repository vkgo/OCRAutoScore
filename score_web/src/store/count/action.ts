import { ADD, SUB } from "./constants"
import { IAction } from "./types"

export function addAction(count: number): IAction {
    return {
        type: ADD,
        count
    }
}

export function subAction(count: number): IAction {
    return {
        type: SUB,
        count
    }
}