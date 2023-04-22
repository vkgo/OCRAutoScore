import {Card, Form, Input, Select, Button, message} from 'antd'
import React, { useState } from 'react'
import './Login.less'
import axios from 'axios'
import qs from 'qs';

const Login: React.FC = (props:any) => {
    const [form] = Form.useForm()
    const[isLogin, setIsLogin] = useState(true)   
    const onFinish = async () => {
        if(!isLogin) {
            loginHttpRequest("register/")
        }
        else {
            let flag: boolean = await loginHttpRequest("login/")
            if(flag) {
                if (form.getFieldValue("identity") === 'student') {
                    window.sessionStorage.setItem("role", "student")
                    window.sessionStorage.setItem("username", form.getFieldValue("username"))
                    props.history.push("student/papers")
                }
                else {
                    window.sessionStorage.setItem("role", "student")
                    window.sessionStorage.setItem("username", form.getFieldValue("username"))
                    props.history.push("teacher/list")
                }
            }
        }
    }
    const loginHttpRequest = async (url: string):Promise<boolean> => {
        let result = await axios.request({
            url,
            method: 'POST',
            data: qs.stringify(form.getFieldsValue()),
            headers: { 'content-type': 'application/x-www-form-urlencoded' },
        })
        // console.log(result)
        if(result.data.msg === 'success'){
            message.success( (isLogin ? "登录": "注册") +"成功")
            return true
        }
        message.error(result.data.msg)
        return false
    }

    const changeState = () => {
        setIsLogin(!isLogin);
        
    }

    return (
        <div className="login_page">
            <Card title={isLogin ? '登录':"注册"} style={{width: '40%'}} extra={<Button type="link" onClick={changeState}>{isLogin ? "注册": "登录"}</Button>}>
            <Form
                labelCol={{ span: 4 }}
                wrapperCol={{ span: 16 }}
                style={{width: '100%'}}
                form={form}
                onFinish= {onFinish}
            >
                    <Form.Item label="用户名" name="username" rules={[{required: true, message: '请输入用户名！'}]}>
                        <Input/>
                    </Form.Item>
                    <Form.Item label="身份" name="identity" rules={[{required: true}]}>
                        <Select placeholder="请选择你的身份">
                            <Select.Option value="student">学生</Select.Option>
                            <Select.Option value="teacher">教师</Select.Option>
                        </Select>
                    </Form.Item>
                    
                    <Form.Item label="密码" name="password" rules={[{required: true, message: '请输入密码！'}]}>
                        <Input.Password/>
                    </Form.Item>
                    {
                        isLogin ? "": (
                            <>
                                <Form.Item label="邮箱" name="email" rules={[{required: true, message: '请输入邮箱！'}]}>
                                    <Input/>
                                </Form.Item>
                                <Form.Item label="学校" name="school" rules={[{required: true, message: '请输入你所在学校名称'}]}>
                                    <Input/>
                                </Form.Item>
                            </>
                        )
                    }
                    
                    <Form.Item wrapperCol={{offset: 8, span: 16}}>
                        <Button type='primary' onClick={() => {form.submit();}}>
                            {isLogin?"登录":"注册"}
                        </Button>
                    </Form.Item>
                </Form> 
            </Card>
        </div>
    )
}

export default Login