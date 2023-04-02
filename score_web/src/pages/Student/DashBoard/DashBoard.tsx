import React from "react"
import { renderRoutes, RouteConfigComponentProps } from "react-router-config"
import './DashBoard.less'
interface DashBoardProps extends RouteConfigComponentProps {}
const DashBoard: React.FC<DashBoardProps> = (props) => {
    console.log(props)
    return (
        <div className="student_dashboard_body">
            {renderRoutes(props.route.routes)}
        </div>
    )
}
export default DashBoard;