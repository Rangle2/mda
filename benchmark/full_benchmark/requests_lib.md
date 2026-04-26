# Python Requests Library — Technical Reference

The requests library is a Python HTTP client library. Its current stable version is 2.31.0. It was created by Kenneth Reitz in 2011. The library is built on top of urllib3 and provides a human-friendly API for making HTTP requests.

## Core Request Methods

The requests library provides seven core HTTP methods as top-level functions: get, post, put, patch, delete, head, and options. Each function accepts a url as its first positional argument. All methods return a Response object.

The get function accepts params as a keyword argument for query string parameters. It sends the params dict as URL-encoded query parameters. Example: requests.get("https://api.example.com/users", params={"page": 1, "limit": 10}) sends to "https://api.example.com/users?page=1&limit=10".

The post function accepts data and json as keyword arguments. When json is provided, the Content-Type header is automatically set to application/json and the dict is serialized. When data is provided, it is form-encoded by default.

## The Session Object

The Session object persists parameters across requests. It maintains cookies, headers, and authentication between requests to the same host. A Session uses connection pooling, which means TCP connections are reused across requests to the same host, reducing latency.

Creating a Session: session = requests.Session(). The session.get, session.post and other methods work identically to the module-level functions but persist state. Sessions should be used as context managers: with requests.Session() as session to ensure connections are properly closed.

Session-level headers are merged with request-level headers. If the same header key exists in both, the request-level value takes precedence.

## The Response Object

Every request returns a Response object. Key attributes: status_code (integer HTTP status), text (response body as string decoded with the detected encoding), content (response body as raw bytes), json() (parses response body as JSON, raises JSONDecodeError if invalid), headers (case-insensitive dict of response headers), url (final URL after redirects), elapsed (timedelta of request duration), history (list of Response objects for redirect chain).

Response.raise_for_status() raises an HTTPError if the status code indicates an error (4xx or 5xx). It does nothing for 2xx and 3xx responses.

Response.ok returns True if status_code is less than 400.

## Authentication

requests supports four built-in authentication schemes. HTTPBasicAuth accepts username and password and encodes them as Base64 in the Authorization header. HTTPDigestAuth uses the HTTP Digest authentication protocol, which involves a challenge-response mechanism and never sends the password in plaintext. HTTPProxyAuth is identical to HTTPBasicAuth but sets the Proxy-Authorization header instead. OAuth1 requires the requests-oauthlib package and is not built into requests itself.

Custom authentication can be implemented by subclassing requests.auth.AuthBase and implementing the __call__ method.

## Timeouts and Retries

The timeout parameter accepts either a single float (total timeout) or a tuple of two floats (connect timeout, read timeout). Example: requests.get(url, timeout=(3.05, 27)) sets a 3.05 second connect timeout and 27 second read timeout. requests raises a ConnectTimeout if the connection cannot be established within the connect timeout. It raises a ReadTimeout if the server does not send data within the read timeout.

requests does not implement automatic retries natively. Retries require urllib3.util.retry.Retry configured through a HTTPAdapter mounted on a Session. The Retry object accepts total (maximum retry count), backoff_factor (sleep time between retries), and status_forcelist (list of HTTP status codes that trigger a retry).

## SSL and Certificates

By default, requests verifies SSL certificates using the certifi bundle. The verify parameter can be set to False to disable verification (insecure) or to a path string pointing to a CA bundle file. The cert parameter accepts a path to a client-side certificate for mutual TLS authentication. It accepts either a path to a combined PEM file or a tuple of (cert, key) paths.

## Streaming

Setting stream=True on a request prevents the response body from being immediately downloaded. The response body can then be iterated with response.iter_content(chunk_size=8192) for binary data or response.iter_lines() for line-by-line text. The connection remains open until response.close() is called or the response is used as a context manager.

## PreparedRequest

Internally, requests converts every request into a PreparedRequest object before sending. A PreparedRequest contains the finalized URL, headers, and body. It can be created manually using session.prepare_request(Request(...)) to inspect or modify the request before sending it with session.send(prepared).

## Common Exceptions

requests.exceptions.ConnectionError is raised when a network problem occurs such as DNS failure or refused connection. requests.exceptions.Timeout is the base class for ConnectTimeout and ReadTimeout. requests.exceptions.HTTPError is raised by raise_for_status() for 4xx and 5xx responses. requests.exceptions.TooManyRedirects is raised when the redirect limit (default 30) is exceeded. All requests exceptions inherit from IOError.
