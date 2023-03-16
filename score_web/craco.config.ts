import path from "path"
import CracoLessPlugin from 'craco-less'
module.exports =  {
    webpack: {
        alias: {
            "@": path.join(__dirname, "src")
        }
    },
    plugins: [
        {
            plugin: CracoLessPlugin,
            options: {
               lessLoaderOptions: {
                lessOptions: {
                    modifyVars: {'@primary-color':'#1677ff'},
                    javasciptEnabled: true
                }
               } 
            }
        }
    ]
}

export{}