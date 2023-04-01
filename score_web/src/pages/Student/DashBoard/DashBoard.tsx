import React from "react"
import { renderRoutes, RouteConfigComponentProps } from "react-router-config"
interface DashBoardProps extends RouteConfigComponentProps {}
const DashBoard: React.FC<DashBoardProps> = (props) => {
    console.log(props)
    return (
        <div>
            {renderRoutes(props.route.routes)}
        </div>
    )
}
export default DashBoard;