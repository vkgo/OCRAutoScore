// store/index.ts
import { createStore } from 'redux'
import reducer from './reducer'

//到合并的reducer类型
import { IRootReducer } from './reducer'

const store = createStore(reducer)
export default store

//导出reducer类型，在组件中： useSelector的时候使用
export type { IRootReducer }
