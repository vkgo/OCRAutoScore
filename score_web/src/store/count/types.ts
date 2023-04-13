export interface IInitCountState {
    count: number
}

export interface IAction {
    type: string
    [index: string]: any
}