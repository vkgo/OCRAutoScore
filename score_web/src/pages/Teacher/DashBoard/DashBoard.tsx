import React from "react"
import { RouteConfigComponentProps, renderRoutes } from "react-router-config"
import { Link } from 'react-router-dom';
import './DashBoard.less'
import { Breadcrumb } from "antd"
interface DashBoardProps extends RouteConfigComponentProps {}
const breadcrumbNameMap: Record<string, string> = {
    '/teacher': '首页',
    '/teacher/list': '查看试卷列表',
    '/teacher/list/add': '添加试卷',
    '/teacher/list/detail': '查看试卷详情',
};
interface BreadcrumbItemInterface {
    title: string | JSX.Element,
    key?: string
}
const DashBoard: React.FC<DashBoardProps> = (props) => {
    console.log('props:', props)
    // 拆分当前路径
    const pathSnippets = props.location.pathname.split('/').filter((i) => i);
    console.log('pathSnippets:', pathSnippets)

    let breadcrumbItems : BreadcrumbItemInterface[] = [];
    for(let i = 0; i < pathSnippets.length; i++) {
        const url = `/${pathSnippets.slice(0, i + 1).join('/')}`;
        if(!(url in breadcrumbNameMap)) continue;
        breadcrumbItems.push({
            key: url,
            title: url === '/teacher' || url==='/teacher/list/detail'? breadcrumbNameMap[url] :<Link to={url}>{breadcrumbNameMap[url]}</Link>,
        });
    };



    return (
        <div className="techer_dashboard_body">
            <div className="header">
                <Breadcrumb items = {breadcrumbItems}/>
                <span className="name">梁老师，你好</span>
            </div>
            {renderRoutes(props.route.routes)}
        </div>
    )
}
export default DashBoard;