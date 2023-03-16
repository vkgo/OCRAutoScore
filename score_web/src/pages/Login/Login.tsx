import {Card, Form, Input, Select, Button} from 'antd'
import React from 'react'
import './Login.less'
const Login: React.FC = () => {
    return (
        <div className="login_page">
            <Card title="登录" style={{width: '40%'}} extra={<a href='#'>注册</a>}>
            <Form
                    labelCol={{ span: 4 }}
                    wrapperCol={{ span: 16 }}
                    style={{width: '100%'}}
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
                    <Form.Item label="邮箱" name="email" rules={[{required: true, message: '请输入邮箱！'}]}>
                        <Input/>
                    </Form.Item>
                    <Form.Item label="密码" name="password" rules={[{required: true, message: '请输入密码！'}]}>
                        <Input/>
                    </Form.Item>
                    <Form.Item wrapperCol={{offset: 8, span: 16}}>
                        <Button type='primary' htmlType='submit'>
                            登录
                        </Button>
                    </Form.Item>
                </Form> 
            </Card>
        </div>
    )
}

export default Login