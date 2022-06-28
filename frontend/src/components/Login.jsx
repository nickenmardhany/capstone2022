import React from "react";
import { useContext } from "react";
import { useState } from "react";

import { UserContext } from "../context/UserContext";
import ErrorMessage from "./ErrorMessage";

const Login = () => {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [errorMessage, setErrorMessage] = useState("");
    const [,setToken] = useContext(UserContext);


    const submitLogin = async () => {
        const requestOptions = {
            method: "POST",
            headers: { "Content-Type": "application/json"},
            body: JSON.stringify({email: email, password:password})

        };

        const response = await fetch("/login", requestOptions);
        const data = await response.json();

        if(!response.ok){
            setErrorMessage(data.detail);
        } 
        else {
            setToken(data.token)
            console.log(data.token)
        }
        


    };
    const handleSubmit = (e) => {
        e.preventDefault();
        if (password.length > 5) {
            submitLogin();
        } else {
            setErrorMessage(
                "Ensure password greater than 5 characters"
            );
        }
    };

    return(
        <div className="column ">
            <form action="" className="box" onSubmit={handleSubmit}>
                <h1 className="title has-text-centered">Login</h1>
        
                <div className="field">
                    <label className="label">Email Address</label>
                    <div className="control">
                        <input type="email" placeholder="Enter email" value={email} onChange={(e)=> setEmail(e.target.value)} className="input" required />
                    </div>
                </div>
                <div className="field">
                    <label className="label">Password</label>
                    <div className="control">
                        <input type="password" placeholder="Enter password" value={password} onChange={(e)=> setPassword(e.target.value)} className="input" required />
                    </div>
                </div>

                <ErrorMessage message={errorMessage}/>
                <br/>
                <button className="button is-primary" type="submit">Login</button>
                
            </form>
        
        </div>
    )
};

export default Login;